import os
import json
import pickle
import numpy as np
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel


# -------------------------------
# Pydantic Models
# -------------------------------
class Chunk(BaseModel):
    doc_id: str
    title: str
    section_path: str
    content: str


class SearchResult(BaseModel):
    doc_id: str
    title: str
    section_path: str
    content: str
    similarity: float


class Metadata(BaseModel):
    chunks: list[Chunk]
    embedding_dim: int
    num_chunks: int
    num_docs: int


# -------------------------------
# Markdown Chunker (no regex)
# -------------------------------
def chunk_markdown(md_text: str, doc_id: str) -> list[Chunk]:
    """
    Chunks a markdown document into sections based on headings (# to ######).
    Each chunk contains a section's content and its full hierarchical path.
    """
    lines = md_text.strip().splitlines()
    if not lines:
        return []

    chunks = []
    current_title = ""
    section_stack = []  # Tracks nested headings: [{"level": 1, "text": "Heading 1"}]
    content_buffer = []

    def save_chunk():
        """Helper to flush the content buffer into a new chunk."""
        if not content_buffer or not section_stack:
            return

        chunk = Chunk(
            doc_id=doc_id,
            title=current_title,
            section_path=" > ".join(s["text"] for s in section_stack),
            content="\n".join(content_buffer).strip(),
        )
        chunks.append(chunk)
        content_buffer.clear()

    for line in lines:
        # Check for line break and stop processing
        if line.strip() == "---":
            break  # Stop chunking entirely, don't include this line

        stripped_line = line.lstrip()
        # Check if the line is a valid heading.
        if stripped_line.startswith("#"):
            level = 0
            while level < len(stripped_line) and stripped_line[level] == "#":
                level += 1

            if (
                1 <= level <= 6
                and len(stripped_line) > level
                and stripped_line[level] == " "
            ):
                heading_text = stripped_line[level + 1 :].strip()
                save_chunk()  # Save content from the previous section

                if level == 1:
                    # Top-level heading becomes the document title
                    current_title = heading_text
                    section_stack.clear()
                else:
                    # Pop deeper headings to match the new heading's level
                    while section_stack and section_stack[-1]["level"] >= level:
                        section_stack.pop()
                    section_stack.append({"level": level, "text": heading_text})
                continue  # Skip adding the heading line to content

        # Normal content line
        content_buffer.append(line)

    save_chunk()  # Flush the very last chunk

    return chunks


# -------------------------------
# Hybrid Vector Store
# -------------------------------
class HybridVectorStore:
    """
    A simple file-based vector store for Markdown documents.
    It chunks documents, embeds them using Sentence-BERT, and supports semantic search.
    """

    def __init__(
        self,
        embeddings_file: str = "vector-store/embeddings.pkl",
        metadata_file: str = "vector-store/metadata.json",
    ) -> None:
        self.embeddings_file = embeddings_file
        self.metadata_file = metadata_file
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[dict] = None
        self.model = SentenceTransformer(
            "sentence-transformers/embeddinggemma-300m-medical"
        )

    def precompute_embeddings(
        self, documents: list[str], doc_ids: Optional[list[str]] = None
    ) -> None:
        """
        Processes and embeds a list of markdown documents and saves the results.
        If doc_ids are not provided, generic IDs will be created.
        """
        print("Chunking and computing embeddings...")
        all_chunks: list[Chunk] = []
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]

        for doc_text, doc_id in zip(documents, doc_ids):
            chunks = chunk_markdown(doc_text, doc_id)
            all_chunks.extend(chunks)

        if not all_chunks:
            print(
                "Warning: No chunks were generated. Skipping embedding computation and saving empty files."
            )
            self.embeddings = np.array([])
            self.metadata = {
                "chunks": [],
                "embedding_dim": 0,
                "num_chunks": 0,
                "num_docs": len(documents),
            }
        else:
            # Concatenate section path with content for embedding
            contents = [f"{c.section_path} {c.content}" for c in all_chunks]
            self.embeddings = self.model.encode(contents, show_progress_bar=True)

            self.metadata = {
                "chunks": [chunk.model_dump() for chunk in all_chunks],
                "embedding_dim": self.embeddings.shape[1],
                "num_chunks": len(all_chunks),
                "num_docs": len(documents),
            }

        with open(self.embeddings_file, "wb") as f:
            pickle.dump(self.embeddings, f)

        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Saved {len(all_chunks)} chunks from {len(documents)} docs.")

    def load_embeddings(self) -> None:
        """Loads pre-computed embeddings and metadata from disk into memory."""
        if not os.path.exists(self.embeddings_file) or not os.path.exists(
            self.metadata_file
        ):
            raise FileNotFoundError(
                "Embedding files not found. Run precompute_embeddings() first to create them."
            )

        with open(self.embeddings_file, "rb") as f:
            self.embeddings = pickle.load(f)

        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)

        if self.metadata is None or self.embeddings is None:
            raise RuntimeError("Embeddings or metadata could not be loaded properly.")

        print(f"Loaded {self.metadata.get('num_chunks', 0)} chunks into memory.")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Performs a semantic search for the most relevant chunks."""
        if self.embeddings is None or self.metadata is None:
            self.load_embeddings()

        if self.embeddings is None or self.metadata is None:
            raise RuntimeError("error")

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_data = self.metadata["chunks"][idx]
            chunk = Chunk(**chunk_data)
            results.append(
                SearchResult(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    section_path=chunk.section_path,
                    content=chunk.content,
                    similarity=float(similarities[idx]),
                )
            )
        return results

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        """Retrieves all chunks belonging to a specific document ID."""
        if self.metadata is None:
            self.load_embeddings()

        if self.metadata is None:
            return []

        chunks = []
        for chunk_data in self.metadata["chunks"]:
            if chunk_data["doc_id"] == doc_id:
                chunks.append(Chunk(**chunk_data))
        return chunks

    def reconstruct_document(self, doc_id: str) -> str:
        """Reconstructs the original markdown document from its stored chunks."""
        if self.metadata is None:
            self.load_embeddings()

        chunks = self.get_document_chunks(doc_id)
        if not chunks:
            return ""

        rebuilt_lines = []
        title = chunks[0].title
        if title:
            rebuilt_lines.append(f"# {title}\n")

        for chunk in chunks:
            path_parts = chunk.section_path.split(" > ")
            # A level 2 heading for the first section, 3 for a nested one, etc.
            heading_level = len(path_parts) + 1
            if len(path_parts) > 0:
                heading = path_parts[-1]
                rebuilt_lines.append(f"{'#' * heading_level} {heading}")
            rebuilt_lines.append(chunk.content)
            rebuilt_lines.append("")  # Add a newline for separation

        return "\n".join(rebuilt_lines).strip()

    def load_documents_from_folder(
        self, folder_path: str = "vector-store/doc"
    ) -> Tuple[list[str], list[str]]:
        """
        Recursively loads all markdown files from a specified folder and its subfolders.

        Args:
            folder_path (str): Path to the folder containing markdown files. Defaults to "vector-store/doc".

        Returns:
            tuple[list[str], list[str]]: A tuple containing (documents, document_ids)
        """
        documents = []
        doc_ids = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

        # Recursively find all .md files in the folder and subfolders
        md_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        if not md_files:
            print(f"No markdown files found in '{folder_path}' (including subfolders)")
            return documents, doc_ids

        for file_path in md_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    # Use relative path without extension as doc_id
                    relative_path = os.path.relpath(file_path, folder_path)
                    doc_id = os.path.splitext(relative_path)[0]
                    # Replace path separators with underscores for doc_id
                    doc_id = doc_id.replace(os.sep, "_")
                    doc_ids.append(doc_id)
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}': {e}")

        print(
            f"Loaded {len(documents)} documents from '{folder_path}' (including subfolders)"
        )
        return documents, doc_ids


def setup_from_folder_example() -> HybridVectorStore:
    """Example of setting up the vector store from markdown files in the doc folder."""
    vector_store = HybridVectorStore()

    # Load documents from the doc folder
    documents, doc_ids = vector_store.load_documents_from_folder()

    # Process and save embeddings
    vector_store.precompute_embeddings(documents, doc_ids)
    return vector_store


if __name__ == "__main__":
    print("--- Setting up vector store ---")
    try:
        print("\n" + "=" * 50)
        print("Loading documents from folder...")
        setup_from_folder_example()
        vector = HybridVectorStore()

        # Perform a search on the documents loaded from the folder
        print("\n--- Searching the vector store (from folder) ---")
        query = "acne first line treatment"
        results = vector.search(query, top_k=5)

        print(f"Query: '{query}'")
        print("Results:")
        for r in results:
            print(
                f"- Document: {r.doc_id}\n"
                f"  Section: {r.title} > {r.section_path}\n"
                f"  Content: '{r.content}' (Similarity: {r.similarity:.3f})\n"
            )

    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
    except RuntimeError as e:
        print(f"An error occurred: {e}")
