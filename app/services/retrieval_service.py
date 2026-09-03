import logging
import re

from sentence_transformers import CrossEncoder

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - graceful fallback if not installed
    BM25Okapi = None

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.config import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


class RetrievalService:

    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    RRF_K = 60

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
    ):
        self._embedder = embedding_service
        self._vector_store = vector_store_service
        self._reranker = None

    def retrieve(
        self,
        question: str,
        top_k: int = RETRIEVAL_TOP_K,
        bucket: str | None = None,
    ) -> list[dict]:
        """
        Find the top-K document chunks most relevant to the given question.

        The primary path is hybrid retrieval: semantic ChromaDB search + BM25 keyword
        search fused via Reciprocal Rank Fusion (RRF), followed by a cross-encoder
        rerank. If hybrid retrieval fails or yields no usable candidates, semantic-only
        retrieval remains available as a fallback.
        """
        if self._vector_store.count() == 0:
            raise RuntimeError(
                "The vector store is empty. "
                "Run the ingestion pipeline first before querying."
            )

        logger.info(
            f"Retrieving chunks for question: '{question[:80]}...' "
            f"| top_k={top_k} | bucket={bucket or 'ALL'}"
        )

        query_vector = self._embedder.embed_one(question)
        candidate_limit = max(top_k * 2, top_k)

        semantic_candidates = self._vector_store.search(
            query_embedding=query_vector,
            top_k=candidate_limit,
            bucket=bucket,
        )

        fused_candidates = semantic_candidates
        try:
            bm25_candidates = self._search_bm25(question, top_k=candidate_limit, bucket=bucket)
            if bm25_candidates:
                fused_candidates = self._rrf_fuse(
                    semantic_candidates,
                    bm25_candidates,
                    k=self.RRF_K,
                )
                logger.info(
                    "Hybrid retrieval used: semantic + BM25 fused via RRF "
                    f"({len(semantic_candidates)} + {len(bm25_candidates)} candidates)"
                )
            else:
                logger.info("BM25 returned no candidates; using semantic-only fallback.")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Hybrid retrieval failed; falling back to semantic-only search. Error: %s",
                exc,
            )
            fused_candidates = semantic_candidates

        chunks = self._rerank_with_cross_encoder(question, fused_candidates, top_k)

        self._log_retrieved_chunks(chunks)

        return chunks

    def _search_bm25(
        self,
        question: str,
        top_k: int,
        bucket: str | None = None,
    ) -> list[dict]:
        """Search the filtered collection using a BM25 keyword ranker."""
        if not question or not question.strip():
            return []

        if BM25Okapi is None:
            logger.warning("rank_bm25 is not installed; skipping BM25 retrieval.")
            return []

        collection = self._vector_store._collection
        where = {"bucket": bucket} if bucket else None

        result = collection.get(where=where, include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not documents:
            return []

        tokenized_docs = [self._tokenize(doc) for doc in documents]
        tokenized_query = self._tokenize(question)
        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        ranked = []
        for idx in ranked_indices:
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            ranked.append(
                {
                    "text": documents[idx],
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "file_type": metadata.get("file_type"),
                    "bucket": metadata.get("bucket"),
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": 1.0,
                    "bm25_score": round(float(scores[idx]), 4),
                }
            )

        return ranked

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _candidate_key(chunk: dict) -> str:
        source = chunk.get("source") or ""
        bucket = chunk.get("bucket") or ""
        page = chunk.get("page") or ""
        chunk_index = chunk.get("chunk_index") or ""
        return f"{bucket}|{source}|{page}|{chunk_index}"

    def _rrf_fuse(
        self,
        semantic_chunks: list[dict],
        bm25_chunks: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """Combine semantic and BM25 candidate lists using Reciprocal Rank Fusion."""
        fused: dict[str, dict] = {}

        for rank, chunk in enumerate(semantic_chunks, start=1):
            key = self._candidate_key(chunk)
            if key not in fused:
                fused[key] = {**chunk}
            fused[key]["rrf_score"] = fused.get(key, {}).get("rrf_score", 0.0) + 1.0 / (k + rank)

        for rank, chunk in enumerate(bm25_chunks, start=1):
            key = self._candidate_key(chunk)
            if key not in fused:
                fused[key] = {**chunk}
            fused[key]["rrf_score"] = fused.get(key, {}).get("rrf_score", 0.0) + 1.0 / (k + rank)
            if "bm25_score" in chunk:
                fused[key]["bm25_score"] = chunk.get("bm25_score")
            if "distance" not in fused[key] and "distance" in chunk:
                fused[key]["distance"] = chunk.get("distance")

        ranked = sorted(fused.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)
        return ranked

    def _rerank_with_cross_encoder(
        self,
        question: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Rerank hybrid candidate list using a local cross-encoder."""
        if not chunks:
            return []

        if self._reranker is None:
            logger.info(f"Loading reranker model: '{self.RERANKER_MODEL}' ...")
            self._reranker = CrossEncoder(self.RERANKER_MODEL)
            logger.info("Reranker model ready.")

        pairs = [(question, chunk.get("text", "")) for chunk in chunks]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = []
        for chunk, score in zip(chunks, scores):
            ranked.append({**chunk, "reranker_score": round(float(score), 4)})

        ranked.sort(key=lambda chunk: chunk["reranker_score"], reverse=True)
        return ranked[:top_k]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _log_retrieved_chunks(self, chunks: list[dict]) -> None:

        if not chunks:
            logger.warning("No chunks were retrieved!")
            return

        logger.info(f"─── Retrieved {len(chunks)} chunk(s) ───")
        for i, chunk in enumerate(chunks):
            logger.info(
                f"  [{i+1}] source={chunk['source']} | "
                f"page={chunk['page']} | "
                f"bucket={chunk['bucket']} | "
                f"distance={chunk.get('distance', 'N/A')} | "
                f"text_preview={chunk['text'][:80].replace(chr(10), ' ')!r}"
            )
        logger.info("─" * 40)
