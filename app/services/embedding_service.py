import logging
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL
logger = logging.getLogger(__name__)

class EmbeddingService:

    def __init__(self, model_name: str=EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: '{model_name}' ...")
        self._model = SentenceTransformer(model_name)
        logger.info(f'Embedding model ready. Vector size: {self._model.get_embedding_dimension()} dimensions')

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning('embed() called with an empty list — returning []')
            return []
        logger.info(f'Embedding {len(texts)} text(s)...')
        vectors = self._model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

# Code update
