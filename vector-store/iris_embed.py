import os
import json
import iris
import numpy as np
import uuid
from typing import Optional, Tuple, List
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from markdown_chunker import chunk_markdown, Chunk


class SearchResult(BaseModel):
    id: str  # Added UUID field
    doc_id: str
    title: str
    section_path: str
    content: str
    similarity: float


class SearchDocumentResult(BaseModel):
    doc_id: str
    full_document: str
    frequency_in_top_k: int
    max_similarity_score: float
    was_tie_breaker: bool
    query: str


class IRISVectorStore:
    def __init__(
        self,
        username: str = "demo",
        password: str = "demo",
        hostname: str = "localhost",
        port: str = "1972",
        namespace: str = "USER",
        table_name: str = "VectorStore.DocumentChunks",
    ) -> None:
        self.username = username
        self.password = password
        self.hostname = hostname
        self.port = port
        self.namespace = namespace
        self.table_name = table_name
        self.connection_string = f"{hostname}:{port}/{namespace}"
        self.conn = None
        self.cursor = None
        self.model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
        self.embedding_dim = 768  # PubMedBERT embedding dimension

    def connect(self) -> None:
        """Establish connection to IRIS database"""
        try:
            self.conn = iris.connect(
                self.connection_string, self.username, self.password
            )
            self.cursor = self.conn.cursor()
            print(f"Connected to IRIS at {self.connection_string}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IRIS: {e}")

    def disconnect(self) -> None:
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Disconnected from IRIS")

    def create_table(self) -> None:
        """Create the vector store table if it doesn't exist"""
        if not self.cursor:
            self.connect()

        # Check if connection was successful
        if not self.cursor:
            raise RuntimeError("Failed to establish database connection")

        # Drop table if exists (for fresh start)
        try:
            self.cursor.execute(f"DROP TABLE {self.table_name}")
            print(f"Dropped existing table {self.table_name}")
        except Exception as e:
            print(f"Note: Could not drop table {self.table_name} (may not exist): {e}")
            pass

        # Create table with UUID as primary key and vector column
        table_definition = f"""(
            id VARCHAR(36) PRIMARY KEY,
            doc_id VARCHAR(255),
            title VARCHAR(500),
            section_path VARCHAR(1000),
            content VARCHAR(5000),
            embedding VECTOR(DOUBLE, {self.embedding_dim})
        )"""

        self.cursor.execute(f"CREATE TABLE {self.table_name} {table_definition}")
        print(f"Created table {self.table_name}")

    def precompute_embeddings(
        self, documents: List[str], doc_ids: Optional[List[str]] = None
    ) -> None:
        """Chunk documents, compute embeddings, and store in IRIS"""
        if not self.cursor:
            self.connect()

        # Check if connection was successful
        if not self.cursor:
            raise RuntimeError("Failed to establish database connection")

        print("Chunking and computing embeddings...")
        all_chunks: List[Chunk] = []
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]

        # Chunk all documents
        for doc_text, doc_id in zip(documents, doc_ids):
            chunks = chunk_markdown(doc_text, doc_id)
            all_chunks.extend(chunks)

        if not all_chunks:
            print("Warning: No chunks were generated.")
            return

        # Compute embeddings for all chunks
        contents = [f"{c.section_path} {c.content}" for c in all_chunks]
        embeddings = self.model.encode(contents, show_progress_bar=True)

        # Prepare data for batch insert with UUID
        sql = f"""
            INSERT INTO {self.table_name}
            (id, doc_id, title, section_path, content, embedding) 
            VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
        """

        data = []
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk_id = str(uuid.uuid4())  # Generate UUID for each chunk
            data.append(
                (
                    chunk_id,
                    chunk.doc_id,
                    chunk.title,
                    chunk.section_path,
                    chunk.content,
                    str(embedding.tolist()),
                )
            )

        # Batch insert all chunks
        print(f"Inserting {len(data)} chunks into IRIS...")
        self.cursor.executemany(sql, data)
        print(
            f"Successfully stored {len(all_chunks)} chunks from {len(documents)} docs."
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for top-k most similar chunks using vector similarity"""
        if not self.cursor:
            self.connect()

        # Check if connection was successful
        if not self.cursor:
            raise RuntimeError("Failed to establish database connection")

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        # SQL query with vector dot product for cosine similarity
        sql = f"""
            SELECT TOP ? id, doc_id, title, section_path, content,
                   VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, DOUBLE)) as similarity
            FROM {self.table_name}
            ORDER BY similarity DESC
        """

        self.cursor.execute(sql, [top_k, str(query_embedding.tolist())])
        rows = self.cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                SearchResult(
                    id=row[0],
                    doc_id=row[1],
                    title=row[2],
                    section_path=row[3],
                    content=row[4],
                    similarity=float(row[5]),
                )
            )

        return results

    def search_document(self, query: str, top_k: int = 5) -> SearchDocumentResult:
        """Search for the most relevant document based on chunk frequency and similarity"""
        if not self.cursor:
            self.connect()

        # Check if connection was successful
        if not self.cursor:
            raise RuntimeError("Failed to establish database connection")

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        # Get top-k chunks
        sql = f"""
            SELECT TOP ? doc_id,
                   VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, DOUBLE)) as similarity
            FROM {self.table_name}
            ORDER BY similarity DESC
        """

        self.cursor.execute(sql, [top_k, str(query_embedding.tolist())])
        rows = self.cursor.fetchall()

        # Calculate frequency and max similarity per document
        doc_frequencies = {}
        doc_max_similarities = {}

        for row in rows:
            doc_id = row[0]
            similarity = float(row[1])

            doc_frequencies[doc_id] = doc_frequencies.get(doc_id, 0) + 1

            if (
                doc_id not in doc_max_similarities
                or similarity > doc_max_similarities[doc_id]
            ):
                doc_max_similarities[doc_id] = similarity

        # Find most frequent document(s)
        max_frequency = max(doc_frequencies.values())
        most_frequent_docs = [
            doc_id for doc_id, freq in doc_frequencies.items() if freq == max_frequency
        ]

        was_tie_breaker = len(most_frequent_docs) > 1

        # Break ties with max similarity
        if was_tie_breaker:
            selected_doc_id = max(
                most_frequent_docs, key=lambda doc_id: doc_max_similarities[doc_id]
            )
        else:
            selected_doc_id = most_frequent_docs[0]

        # Reconstruct full document
        full_document = self.reconstruct_document(selected_doc_id)

        return SearchDocumentResult(
            doc_id=selected_doc_id,
            full_document=full_document,
            frequency_in_top_k=doc_frequencies[selected_doc_id],
            max_similarity_score=doc_max_similarities[selected_doc_id],
            was_tie_breaker=was_tie_breaker,
            query=query,
        )

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """Retrieve all chunks for a specific document"""
        if not self.cursor:
            self.connect()

        # Check if connection was successful
        if not self.cursor:
            raise RuntimeError("Failed to establish database connection")

        sql = f"""
            SELECT doc_id, title, section_path, content
            FROM {self.table_name}
            WHERE doc_id = ?
        """

        self.cursor.execute(sql, [doc_id])
        rows = self.cursor.fetchall()

        chunks = []
        for row in rows:
            chunks.append(
                Chunk(doc_id=row[0], title=row[1], section_path=row[2], content=row[3])
            )

        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Retrieve a specific chunk by its UUID"""
        if not self.cursor:
            self.connect()

        if not self.cursor:
            raise RuntimeError("Failed to establish database connection")

        sql = f"""
            SELECT id, doc_id, title, section_path, content
            FROM {self.table_name}
            WHERE id = ?
        """

        self.cursor.execute(sql, [chunk_id])
        row = self.cursor.fetchone()

        if row:
            return SearchResult(
                id=row[0],
                doc_id=row[1],
                title=row[2],
                section_path=row[3],
                content=row[4],
                similarity=0.0,  # No similarity score for direct lookup
            )

        return None

    def reconstruct_document(self, doc_id: str) -> str:
        """Reconstruct the full document from its chunks"""
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
            if len(path_parts) > 0 and path_parts[0]:
                heading = path_parts[-1]
                rebuilt_lines.append(f"{'#' * heading_level} {heading}")
            rebuilt_lines.append(chunk.content)
            rebuilt_lines.append("")

        return "\n".join(rebuilt_lines).strip()

    def load_documents_from_folder(
        self, folder_path: str = "vector-store/doc"
    ) -> Tuple[List[str], List[str]]:
        """Load all markdown documents from a folder"""
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
            print(f"No markdown files found in '{folder_path}'")
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

        print(f"Loaded {len(documents)} documents from '{folder_path}'")
        return documents, doc_ids


def setup_from_folder_example() -> IRISVectorStore:
    """Example: Setup vector store from folder of markdown files"""
    vector_store = IRISVectorStore()
    vector_store.connect()
    vector_store.create_table()

    documents, doc_ids = vector_store.load_documents_from_folder()
    vector_store.precompute_embeddings(documents, doc_ids)

    return vector_store


if __name__ == "__main__":
    print("--- Setting up IRIS vector store ---")
    try:
        # Setup and populate the vector store
        print("\nLoading documents from folder...")
        vector_store = setup_from_folder_example()

        # Perform a search
        print("\n--- Searching the vector store ---")
        query = "avoid medication g6pd"

        # Search for individual chunks
        print(f"\nQuery: '{query}'")
        print("\nTop chunks:")
        chunk_results = vector_store.search(query, top_k=5)
        for i, result in enumerate(chunk_results, 1):
            print(f"\n{i}. ID: {result.id}")
            print(f"   Doc: {result.doc_id}")
            print(f"   Section: {result.section_path}")
            print(f"   Similarity: {result.similarity:.4f}")
            print(f"   Content: {result.content[:200]}...")

        # Example: Get a specific chunk by UUID
        if chunk_results:
            first_chunk_id = chunk_results[0].id
            print(f"\n\n--- Retrieving chunk by ID ---")
            specific_chunk = vector_store.get_chunk_by_id(first_chunk_id)
            if specific_chunk:
                print(f"Retrieved chunk ID: {specific_chunk.id}")
                print(f"Content: {specific_chunk.content[:200]}...")

        # Search for best matching document
        print("\n\n--- Document-level search ---")
        doc_result = vector_store.search_document(query, top_k=5)
        print(f"Best matching document: {doc_result.doc_id}")
        print(f"Frequency in top-{5}: {doc_result.frequency_in_top_k}")
        print(f"Max similarity: {doc_result.max_similarity_score:.4f}")
        print(f"Tie breaker used: {doc_result.was_tie_breaker}")

        # Clean up
        vector_store.disconnect()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
