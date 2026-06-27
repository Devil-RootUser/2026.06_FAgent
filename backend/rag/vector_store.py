from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config.settings import PROJECT_ROOT, load_rag_config


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class SearchHit:
    doc_id: str
    title: str
    source: str
    chunk_id: str
    score: float
    text: str


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return f"PDF document placeholder: {path.name}. Install a PDF parser or upload txt/md notes for full text indexing."
    return path.read_text(encoding="utf-8", errors="ignore")


class KnowledgeStore:
    """EquiMind-inspired RAG store with Chroma config and a reliable keyword fallback."""

    def __init__(self) -> None:
        self.config = load_rag_config()
        self.data_path = PROJECT_ROOT / str(self.config.get("data_path", "data/knowledge"))
        self.persist_dir = PROJECT_ROOT / str(self.config["vector_store"].get("persist_directory", "outputs/chroma_db"))
        self.index_path = self.persist_dir / "keyword_index.json"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Any] = {"documents": {}, "chunks": []}
        self._load_index()
        self._chroma_status = self._probe_chroma()
        if not self._index.get("documents"):
            self.ingest_directory()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": self.config["vector_store"].get("provider", "keyword"),
            "mode": "chroma" if self._chroma_status["enabled"] and self._chroma_status["available"] else "keyword",
            "chroma": self._chroma_status,
            "persist_directory": str(self.persist_dir),
            "collection_name": self.config["vector_store"].get("collection_name", "fagent_knowledge"),
            "document_count": len(self._index["documents"]),
            "chunk_count": len(self._index["chunks"]),
            "chunk_size": self.config["splitter"].get("chunk_size", 700),
            "chunk_overlap": self.config["splitter"].get("chunk_overlap", 90),
        }

    def _probe_chroma(self) -> dict[str, Any]:
        enabled = bool(self.config["vector_store"].get("use_chroma", False))
        if not enabled:
            return {"enabled": False, "available": False, "reason": "RAG_USE_CHROMA is disabled"}
        try:
            import chromadb  # type: ignore  # noqa: F401

            return {"enabled": True, "available": True, "reason": "chromadb import succeeded"}
        except Exception as exc:
            return {"enabled": True, "available": False, "reason": str(exc)}

    def _load_index(self) -> None:
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _split_text(self, text: str) -> list[str]:
        size = int(self.config["splitter"].get("chunk_size", 700))
        overlap = int(self.config["splitter"].get("chunk_overlap", 90))
        if len(text) <= size:
            return [text.strip()] if text.strip() else []
        chunks = []
        step = max(1, size - overlap)
        for start in range(0, len(text), step):
            chunk = text[start : start + size].strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(text):
                break
        return chunks

    def ingest_text(self, filename: str, text: str, persist_file: bool = True, source_path: str | None = None) -> dict[str, Any]:
        raw = text.encode("utf-8")
        digest = _md5_bytes(raw)
        existing = self._index["documents"].get(digest)
        if existing:
            return {"doc_id": digest, "status": "duplicate", **existing}
        safe_name = Path(filename).name or f"{digest}.txt"
        target = self.data_path / safe_name
        if persist_file:
            if target.exists():
                target = self.data_path / f"{digest[:8]}_{safe_name}"
            target.write_text(text, encoding="utf-8")
            source = str(target.relative_to(PROJECT_ROOT))
        else:
            source = source_path or str(target.relative_to(PROJECT_ROOT))
        chunks = self._split_text(text)
        document = {
            "doc_id": digest,
            "filename": safe_name,
            "source": source,
            "md5": digest,
            "created_at": time.time(),
            "chunk_count": len(chunks),
            "size": len(raw),
        }
        self._index["documents"][digest] = document
        for idx, chunk in enumerate(chunks):
            self._index["chunks"].append(
                {
                    "doc_id": digest,
                    "chunk_id": f"{digest}:{idx}",
                    "title": safe_name,
                    "source": document["source"],
                    "text": chunk,
                    "tokens": _tokens(chunk),
                }
            )
        self._save_index()
        return {"status": "indexed", **document}

    def ingest_file(self, path: Path) -> dict[str, Any]:
        allowed = {f".{x.lower().lstrip('.')}" for x in self.config.get("allowed_file_types", [])}
        if path.suffix.lower() not in allowed:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return self.ingest_text(path.name, _read_text_file(path), persist_file=False, source_path=str(path.relative_to(PROJECT_ROOT)))

    def ingest_directory(self) -> dict[str, Any]:
        allowed = {f".{x.lower().lstrip('.')}" for x in self.config.get("allowed_file_types", [])}
        indexed = 0
        duplicates = 0
        for path in sorted(self.data_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed and path != self.index_path:
                result = self.ingest_file(path)
                if result["status"] == "indexed":
                    indexed += 1
                else:
                    duplicates += 1
        return {"indexed": indexed, "duplicates": duplicates, "status": self.status}

    def list_documents(self) -> list[dict[str, Any]]:
        return sorted(self._index["documents"].values(), key=lambda x: x.get("created_at", 0), reverse=True)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = int(top_k or self.config.get("top_k", 4))
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        hits: list[SearchHit] = []
        total_docs = max(1, len(self._index["chunks"]))
        df: dict[str, int] = {}
        for chunk in self._index["chunks"]:
            for token in set(chunk.get("tokens", [])):
                if token in q_set:
                    df[token] = df.get(token, 0) + 1
        for chunk in self._index["chunks"]:
            tokens = chunk.get("tokens", [])
            if not tokens:
                continue
            score = 0.0
            for token in q_tokens:
                tf = tokens.count(token) / len(tokens)
                idf = math.log((1 + total_docs) / (1 + df.get(token, 0))) + 1
                score += tf * idf
            if score > 0:
                hits.append(
                    SearchHit(
                        doc_id=chunk["doc_id"],
                        title=chunk["title"],
                        source=chunk["source"],
                        chunk_id=chunk["chunk_id"],
                        score=score,
                        text=chunk["text"],
                    )
                )
        hits.sort(key=lambda x: x.score, reverse=True)
        return [hit.__dict__ for hit in hits[:top_k]]

    def delete_document(self, doc_id: str) -> bool:
        doc = self._index["documents"].pop(doc_id, None)
        self._index["chunks"] = [chunk for chunk in self._index["chunks"] if chunk.get("doc_id") != doc_id]
        self._save_index()
        if doc:
            path = PROJECT_ROOT / doc.get("source", "")
            if path.exists() and path.is_file():
                path.unlink()
        return bool(doc)


_STORE: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _STORE
    if _STORE is None:
        _STORE = KnowledgeStore()
    return _STORE

