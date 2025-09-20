import os
import json
import pickle
import numpy as np
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel


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


class SearchDocumentResult(BaseModel):
    doc_id: str
    full_document: str
    frequency_in_top_k: int
    max_similarity_score: float
    was_tie_breaker: bool
    query: str


def chunk_markdown(md_text: str, doc_id: str) -> list[Chunk]:
    lines = md_text.strip().splitlines()
    if not lines:
        return []

    chunks = []
    current_title = ""
    section_stack = []
    content_buffer = []

    def save_chunk():
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
        if line.strip() == "---":
            break

        stripped_line = line.lstrip()
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
                save_chunk()

                if level == 1:
                    current_title = heading_text
                    section_stack.clear()
                else:
                    while section_stack and section_stack[-1]["level"] >= level:
                        section_stack.pop()
                    section_stack.append({"level": level, "text": heading_text})
                continue

        content_buffer.append(line)

    save_chunk()

    return chunks


class HybridVectorStore:
    def __init__(
        self,
        embeddings_file: str = "vector-store/embeddings.pkl",
        metadata_file: str = "vector-store/metadata.json",
    ) -> None:
        self.embeddings_file = embeddings_file
        self.metadata_file = metadata_file
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[dict] = None
        self.model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")

    def precompute_embeddings(
        self, documents: list[str], doc_ids: Optional[list[str]] = None
    ) -> None:
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

    def search_document(self, query: str, top_k: int = 5) -> SearchDocumentResult:
        if self.embeddings is None or self.metadata is None:
            self.load_embeddings()

        if self.embeddings is None or self.metadata is None:
            raise RuntimeError("Embeddings or metadata could not be loaded properly.")

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        doc_frequencies = {}
        doc_max_similarities = {}

        for idx in top_indices:
            chunk_data = self.metadata["chunks"][idx]
            doc_id = chunk_data["doc_id"]
            similarity = float(similarities[idx])
            doc_frequencies[doc_id] = doc_frequencies.get(doc_id, 0) + 1

            if (
                doc_id not in doc_max_similarities
                or similarity > doc_max_similarities[doc_id]
            ):
                doc_max_similarities[doc_id] = similarity

        max_frequency = max(doc_frequencies.values())
        most_frequent_docs = [
            doc_id for doc_id, freq in doc_frequencies.items() if freq == max_frequency
        ]

        was_tie_breaker = len(most_frequent_docs) > 1

        if was_tie_breaker:
            selected_doc_id = max(
                most_frequent_docs, key=lambda doc_id: doc_max_similarities[doc_id]
            )
        else:
            selected_doc_id = most_frequent_docs[0]

        full_document = self.reconstruct_document(selected_doc_id)

        return SearchDocumentResult(
            doc_id=selected_doc_id,
            full_document=full_document,
            frequency_in_top_k=doc_frequencies[selected_doc_id],
            max_similarity_score=doc_max_similarities[selected_doc_id],
            was_tie_breaker=was_tie_breaker,
            query=query,
        )

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
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
            heading_level = len(path_parts) + 1
            if len(path_parts) > 0:
                heading = path_parts[-1]
                rebuilt_lines.append(f"{'#' * heading_level} {heading}")
            rebuilt_lines.append(chunk.content)
            rebuilt_lines.append("")

        return "\n".join(rebuilt_lines).strip()

    def load_documents_from_folder(
        self, folder_path: str = "vector-store/doc"
    ) -> Tuple[list[str], list[str]]:
        documents = []
        doc_ids = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

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
                    relative_path = os.path.relpath(file_path, folder_path)
                    doc_id = os.path.splitext(relative_path)[0]
                    doc_id = doc_id.replace(os.sep, "_")
                    doc_ids.append(doc_id)
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}': {e}")

        print(
            f"Loaded {len(documents)} documents from '{folder_path}' (including subfolders)"
        )
        return documents, doc_ids


def setup_from_folder_example() -> HybridVectorStore:
    vector_store = HybridVectorStore()
    documents, doc_ids = vector_store.load_documents_from_folder()
    vector_store.precompute_embeddings(documents, doc_ids)
    return vector_store


if __name__ == "__main__":
    print("--- Setting up vector store ---")
    try:
        print("\n" + "=" * 50)
        print("Loading documents from folder...")
        setup_from_folder_example()
        vector = HybridVectorStore()

        print("\n--- Searching the vector store (from folder) ---")
        query = "avoid medication g6pd"
        results = vector.search_document(query, top_k=5)

        print(f"Query: '{query}'")
        print("Results:")
        print(results)

    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
    except RuntimeError as e:
        print(f"An error occurred: {e}")
