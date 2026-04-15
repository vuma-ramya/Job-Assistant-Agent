import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os

# Singleton pattern — one client, one model
_client: Optional[chromadb.Client] = None
_embedder: Optional[SentenceTransformer] = None
_collection = None

COLLECTION_NAME = "resumes"
PERSIST_DIR = "./chroma_db"


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Small, fast model — no GPU needed
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def store_resume(session_id: str, chunks: List[str]) -> None:
    """Embed resume chunks and upsert into ChromaDB."""
    collection = _get_collection()
    embedder = _get_embedder()

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"session_id": session_id, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def query_resume(session_id: str, query: str, top_k: int = 5) -> List[str]:
    """Retrieve top-k most relevant resume chunks for a given query."""
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"session_id": session_id},
    )

    if results and results["documents"]:
        return results["documents"][0]
    return []


def delete_session(session_id: str) -> None:
    """Remove all resume chunks for a session (cleanup)."""
    collection = _get_collection()
    results = collection.get(where={"session_id": session_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
