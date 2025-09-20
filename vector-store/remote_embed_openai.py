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
from markdown_chunker import chunk_markdown, Chunk

load_dotenv()

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


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        rate_limit_delay: float = 0.1,
    ):
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("EMBEDDING_URL", ""))
        self.model = model
        self.batch_size = min(batch_size, 2048)
        self.rate_limit_delay = rate_limit_delay

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])

        all_embeddings = []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        iterator = tqdm(batches, desc="Embedding batches") if show_progress else batches

        for batch in iterator:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"},
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                print(f"Error embedding batch: {e}")
                raise

        return np.array(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
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


class HybridVectorStore:
    def __init__(
        self,
        embeddings_file: str = "vector-store/embeddings.pkl",
        metadata_file: str = "vector-store/metadata.json",
        openai_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_model: str = os.getenv("EMBEDDING_MODEL", ""),
        batch_size: int = 100,
        rate_limit_delay: float = 0.1,
    ) -> None:
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

        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)

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
        print(
            f"Embedding model used: {self.metadata.get('embedding_model', 'unknown')}"
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.embeddings is None or self.metadata is None:
            self.load_embeddings()

        if self.embeddings is None or self.metadata is None:
            raise RuntimeError("Failed to load embeddings or metadata.")

        print(f"Embedding query: '{query}'...")
        query_embedding = self.embedder.embed_single(query).reshape(1, -1)

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
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set your EMBEDDING_API_KEY environment variable or pass it to the constructor."
        )

    vector_store = HybridVectorStore(
        openai_api_key=api_key,
        batch_size=50,
        rate_limit_delay=0.2,
    )

    documents, doc_ids = vector_store.load_documents_from_folder()

    vector_store.precompute_embeddings(documents, doc_ids)
    return vector_store


if __name__ == "__main__":
    print("--- Setting up OpenAI-based vector store ---")
    try:
        print("\n" + "=" * 50)
        print("Loading documents from folder...")
        vector_store = setup_from_folder_example()

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
