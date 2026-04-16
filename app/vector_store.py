"""
Vector store — hash-based embeddings. No model download. Instant startup.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Optional
import hashlib
import math

_client = None
_collection = None
COLLECTION_NAME = "resumes"
PERSIST_DIR = "./chroma_db"

def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    return _client

def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _collection

def _embed(texts: List[str], dim: int = 384) -> List[List[float]]:
    results = []
    for text in texts:
        vec = [0.0] * dim
        words = text.lower().split()
        for word in words:
            h1 = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            h2 = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dim
            vec[h1] += 1.0
            vec[h2] += 0.5
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        results.append([x / norm for x in vec])
    return results

def store_resume(session_id: str, chunks: List[str]) -> None:
    collection = _get_collection()
    embeddings = _embed(chunks)
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"session_id": session_id, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

def query_resume(session_id: str, query: str, top_k: int = 5) -> List[str]:
    try:
        collection = _get_collection()
        query_embedding = _embed([query])[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where={"session_id": session_id})
        if results and results["documents"]:
            return results["documents"][0]
    except Exception:
        pass
    return []

def delete_session(session_id: str) -> None:
    try:
        collection = _get_collection()
        results = collection.get(where={"session_id": session_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass
