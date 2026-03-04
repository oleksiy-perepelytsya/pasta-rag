import asyncio
import logging
import socket
from typing import Any

import chromadb
import google.generativeai as genai
import httpx

from config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"


class ChromaStore:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client: chromadb.HttpClient | None = None
        genai.configure(api_key=settings.gemini_api_key)

    async def initialize(self) -> None:
        try:
            try:
                resolved = socket.gethostbyname(self.host)
                logger.info(f"ChromaDB DNS: {self.host} -> {resolved}:{self.port}")
            except Exception:
                logger.warning(f"ChromaDB DNS resolve failed for {self.host}")

            # Health-check retry (6 attempts, 1s apart)
            async with httpx.AsyncClient(timeout=2) as client:
                ok = False
                for _ in range(6):
                    for path in ("/api/v2/health", "/api/v1/heartbeat"):
                        try:
                            resp = await client.get(f"http://{self.host}:{self.port}{path}")
                            if resp.status_code < 500:
                                ok = True
                                break
                        except Exception:
                            pass
                    if ok:
                        break
                    await asyncio.sleep(1)
                if not ok:
                    logger.warning("ChromaDB health check did not succeed; proceeding anyway")

            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            logger.info(f"ChromaDB connected at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB v2 requires all metadata values to be primitive JSON types."""
        result: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                result[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    async def _embed(self, text: str) -> list[float]:
        try:
            result = await genai.embed_content_async(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            emb = result.get("embedding") if isinstance(result, dict) else getattr(result, "embedding", None)
            return [float(x) for x in (emb or [])]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return []

    async def _embed_query(self, text: str) -> list[float]:
        try:
            result = await genai.embed_content_async(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query",
            )
            emb = result.get("embedding") if isinstance(result, dict) else getattr(result, "embedding", None)
            return [float(x) for x in (emb or [])]
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}", exc_info=True)
            return []

    def _get_collection(self, name: str):
        if not self.client:
            raise RuntimeError("ChromaStore not initialized")
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            if "_type" in str(e):
                logger.warning(f"Collection metadata mismatch, recreating: {name}")
                try:
                    self.client.delete_collection(name=name)
                except Exception:
                    pass
                return self.client.create_collection(
                    name=name, metadata={"hnsw:space": "cosine"}
                )
            raise

    def _session_collection(self, telegram_id: int):
        return self._get_collection(f"sessions_{telegram_id}")

    def _kb_collection(self):
        return self._get_collection("knowledge_base")

    # ------------------------------------------------------------------
    # Sessions (per-user namespace)
    # ------------------------------------------------------------------

    async def add_session(
        self,
        telegram_id: int,
        session_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        embedding = await self._embed(text)
        if not embedding:
            logger.warning(f"Skipping session vectorization — no embedding for {session_id}")
            return
        col = self._session_collection(telegram_id)
        col.add(
            ids=[session_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[self._sanitize_metadata(metadata)],
        )
        logger.info(f"Session {session_id} stored in ChromaDB for user {telegram_id}")

    async def query_sessions(
        self, telegram_id: int, query: str, n_results: int = 3
    ) -> list[dict]:
        embedding = await self._embed_query(query)
        if not embedding:
            return []
        try:
            col = self._session_collection(telegram_id)
            count = col.count()
            if count == 0:
                return []
            results = col.query(
                query_embeddings=[embedding],
                n_results=min(n_results, count),
            )
        except Exception as e:
            logger.error(f"Session query failed: {e}", exc_info=True)
            return []

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            score = 1 - dists[i] if i < len(dists) else 0.0
            hits.append({"content": doc, "metadata": meta, "score": score})
        return hits

    async def count_session_embeddings(self, telegram_id: int) -> int:
        try:
            return self._session_collection(telegram_id).count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Knowledge base (shared)
    # ------------------------------------------------------------------

    async def add_chunks(
        self,
        doc_id: str,
        filename: str,
        source_type: str,
        chunks: list[str],
    ) -> int:
        if not chunks:
            return 0
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = []
        for chunk in chunks:
            emb = await self._embed(chunk)
            embeddings.append(emb)

        # Filter out any chunks that failed to embed
        valid = [(i, e) for i, e in enumerate(embeddings) if e]
        if not valid:
            logger.warning(f"No valid embeddings for document {doc_id}")
            return 0

        valid_ids = [ids[i] for i, _ in valid]
        valid_docs = [chunks[i] for i, _ in valid]
        valid_embs = [e for _, e in valid]
        valid_metas = [
            self._sanitize_metadata({
                "doc_id": doc_id,
                "filename": filename,
                "source_type": source_type,
                "chunk_index": i,
            })
            for i, _ in valid
        ]

        col = self._kb_collection()
        col.add(
            ids=valid_ids,
            documents=valid_docs,
            embeddings=valid_embs,
            metadatas=valid_metas,
        )
        logger.info(f"Stored {len(valid_ids)} chunks for document {doc_id}")
        return len(valid_ids)

    async def query_knowledge_base(self, query: str, n_results: int = 3) -> list[dict]:
        embedding = await self._embed_query(query)
        if not embedding:
            return []
        try:
            col = self._kb_collection()
            count = col.count()
            if count == 0:
                return []
            results = col.query(
                query_embeddings=[embedding],
                n_results=min(n_results, count),
            )
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}", exc_info=True)
            return []

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            score = 1 - dists[i] if i < len(dists) else 0.0
            hits.append({"content": doc, "metadata": meta, "score": score})
        return hits

    async def count_kb_embeddings(self) -> int:
        try:
            return self._kb_collection().count()
        except Exception:
            return 0
