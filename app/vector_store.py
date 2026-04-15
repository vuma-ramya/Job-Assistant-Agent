import chromadb
from chromadb.config import Settings
from typing import List, Optional
import os
import httpx

_client = None
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


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _embed(texts: List[str]) -> List[List[float]]:
    """Get embeddings using Groq-compatible or OpenAI embeddings API."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Use OpenAI embeddings (works with both OpenAI and as fallback)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")

    # Always use OpenAI embeddings endpoint (cheap, fast, no download)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_key}"},
            json={"input": texts, "model": "text-embedding-3-small"},
            timeout=30,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    # Fallback: simple TF-IDF style hash embeddings (no API needed)
    # Works with Groq-only setup
    import hashlib
    import math

    def hash_embed(text: str, dim: int = 384) -> List[float]:
        vec = [0.0] * dim
        words = text.lower().split()
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    return [hash_embed(t) for t in texts]


def store_resume(session_id: str, chunks: List[str]) -> None:
    collection = _get_collection()
    embeddings = _embed(chunks)
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"session_id": session_id, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def query_resume(session_id: str, query: str, top_k: int = 5) -> List[str]:
    collection = _get_collection()
    query_embedding = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},
    )
    if results and results["documents"]:
        return results["documents"][0]
    return []


def delete_session(session_id: str) -> None:
    collection = _get_collection()
    results = collection.get(where={"session_id": session_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
