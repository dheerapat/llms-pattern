from pydantic import BaseModel
from markdown_chunker import chunk_markdown, Chunk


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
