import os
import json
import pickle
import numpy as np
from typing import Optional, Tuple, List
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


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
    embedding_model: str


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
    section_stack = []
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
# OpenAI Embedding Helper
# -------------------------------
class OpenAIEmbedder:
    """Helper class to handle OpenAI embeddings with rate limiting and batching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var.
            model: Embedding model to use. Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
            batch_size: Number of texts to embed in each API call (max 2048 for OpenAI)
            rate_limit_delay: Delay between API calls to avoid rate limiting
        """
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("EMBEDDING_URL", ""))
        self.model = model
        self.batch_size = min(batch_size, 2048)  # OpenAI's max batch size
        self.rate_limit_delay = rate_limit_delay

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of texts using OpenAI's embedding API.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        # Process in batches
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        iterator = tqdm(batches, desc="Embedding batches") if show_progress else batches

        for batch in iterator:
            try:
                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"},
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Rate limiting delay
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                print(f"Error embedding batch: {e}")
                raise

        return np.array(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"},
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise


# -------------------------------
# Hybrid Vector Store with OpenAI
# -------------------------------
class HybridVectorStore:
    """
    A simple file-based vector store for Markdown documents using OpenAI embeddings.
    It chunks documents, embeds them using OpenAI's API, and supports semantic search.
    """

    def __init__(
        self,
        embeddings_file: str = "vector-store/embeddings.pkl",
        metadata_file: str = "vector-store/metadata.json",
        openai_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_model: str = os.getenv("EMBEDDING_MODEL", ""),
        batch_size: int = 100,
        rate_limit_delay: float = 0.1,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            embeddings_file: Path to store embeddings pickle file
            metadata_file: Path to store metadata JSON file
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            embedding_model: OpenAI embedding model to use
            batch_size: Batch size for API calls
            rate_limit_delay: Delay between API calls
        """
        self.embeddings_file = embeddings_file
        self.metadata_file = metadata_file
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[dict] = None
        self.embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model=embedding_model,
            batch_size=batch_size,
            rate_limit_delay=rate_limit_delay,
        )
        self.embedding_model = embedding_model

    def precompute_embeddings(
        self, documents: list[str], doc_ids: Optional[list[str]] = None
    ) -> None:
        """
        Processes and embeds a list of markdown documents and saves the results.
        If doc_ids are not provided, generic IDs will be created.
        """
        print("Chunking documents...")
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
                "embedding_model": self.embedding_model,
            }
        else:
            print(f"Computing embeddings for {len(all_chunks)} chunks using OpenAI...")

            # Concatenate section path with content for embedding
            contents = [f"{c.section_path} {c.content}" for c in all_chunks]

            # Use OpenAI to compute embeddings
            self.embeddings = self.embedder.embed_texts(contents, show_progress=True)

            self.metadata = {
                "chunks": [chunk.model_dump() for chunk in all_chunks],
                "embedding_dim": self.embeddings.shape[1],
                "num_chunks": len(all_chunks),
                "num_docs": len(documents),
                "embedding_model": self.embedding_model,
            }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)

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
        print(
            f"Embedding model used: {self.metadata.get('embedding_model', 'unknown')}"
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Performs a semantic search for the most relevant chunks."""
        if self.embeddings is None or self.metadata is None:
            self.load_embeddings()

        if self.embeddings is None or self.metadata is None:
            raise RuntimeError("Failed to load embeddings or metadata.")

        # Embed the query using OpenAI
        print(f"Embedding query: '{query}'...")
        query_embedding = self.embedder.embed_single(query).reshape(1, -1)

        # Calculate similarities
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
    # Make sure to set your OpenAI API key
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set your EMBEDDING_API_KEY environment variable or pass it to the constructor."
        )

    vector_store = HybridVectorStore(
        openai_api_key=api_key,
        batch_size=50,  # Conservative batch size
        rate_limit_delay=0.2,  # 200ms delay between requests
    )

    # Load documents from the doc folder
    documents, doc_ids = vector_store.load_documents_from_folder()

    # Process and save embeddings
    vector_store.precompute_embeddings(documents, doc_ids)
    return vector_store


if __name__ == "__main__":
    print("--- Setting up OpenAI-based vector store ---")
    try:
        print("\n" + "=" * 50)
        print("Loading documents from folder...")
        vector_store = setup_from_folder_example()

        # Perform a search on the documents loaded from the folder
        print("\n--- Searching the vector store (OpenAI embeddings) ---")
        query = "genital herpes management"
        results = vector_store.search(query, top_k=5)

        print(f"Query: '{query}'")
        print("Results:")
        for r in results:
            print(
                f"- Document: {r.doc_id}\n"
                f"  Section: {r.title} > {r.section_path}\n"
                f"  Content: '{r.content[:200]}...' (Similarity: {r.similarity:.3f})\n"
            )

    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
