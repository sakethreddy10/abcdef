import logging
from sentence_transformers import CrossEncoder
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.config import RETRIEVAL_TOP_K
logger = logging.getLogger(__name__)

class RetrievalService:
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-2-v2'

    def __init__(self, embedding_service: EmbeddingService, vector_store_service: VectorStoreService):
        self._embedder = embedding_service
        self._vector_store = vector_store_service
        self._reranker = None

    def retrieve(self, question: str, top_k: int=RETRIEVAL_TOP_K, bucket: str | None=None) -> list[dict]:
        if self._vector_store.count() == 0:
            raise RuntimeError('The vector store is empty. Run the ingestion pipeline first before querying.')
        logger.info(f"Retrieving chunks for question: '{question[:80]}...' | top_k={top_k} | bucket={bucket or 'ALL'}")
        query_vector = self._embedder.embed_one(question)
        vector_chunks = self._vector_store.search(query_embedding=query_vector, top_k=max(top_k * 2, top_k), bucket=bucket)
        chunks = self._rerank_with_cross_encoder(question, vector_chunks, top_k)
        self._log_retrieved_chunks(chunks)
        return chunks

    def _rerank_with_cross_encoder(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        if self._reranker is None:
            logger.info(f"Loading reranker model: '{self.RERANKER_MODEL}' ...")
            self._reranker = CrossEncoder(self.RERANKER_MODEL)
            logger.info('Reranker model ready.')
        pairs = [(question, chunk.get('text', '')) for chunk in chunks]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = []
        for chunk, score in zip(chunks, scores):
            ranked.append({**chunk, 'reranker_score': round(float(score), 4)})
        ranked.sort(key=lambda chunk: chunk['reranker_score'], reverse=True)
        return ranked[:top_k]

    def _log_retrieved_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            logger.warning('No chunks were retrieved!')
            return
        logger.info(f'─── Retrieved {len(chunks)} chunk(s) ───')
        for i, chunk in enumerate(chunks):
            logger.info(f"  [{i + 1}] source={chunk['source']} | page={chunk['page']} | bucket={chunk['bucket']} | distance={chunk.get('distance', 'N/A')} | text_preview={chunk['text'][:80].replace(chr(10), ' ')!r}")
        logger.info('─' * 40)
