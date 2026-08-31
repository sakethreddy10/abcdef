import logging
import hashlib
import chromadb
from chromadb.config import Settings
from app.config import CHROMA_PATH, CHROMA_COLLECTION
logger = logging.getLogger(__name__)

class VectorStoreService:

    def __init__(self, persist_directory: str=CHROMA_PATH, collection_name: str=CHROMA_COLLECTION):
        logger.info(f"Connecting to ChromaDB at '{persist_directory}' (collection: '{collection_name}')")
        self._client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})
        doc_count = self._collection.count()
        logger.info(f"Collection '{collection_name}' ready. Currently holds {doc_count} chunk(s).")

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f'chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length.')
        if not chunks:
            logger.warning('add_chunks() called with empty list — nothing stored.')
            return 0
        ids = []
        documents = []
        metadatas = []
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{self.document_id(chunk['bucket'], chunk['source'])}_p{chunk['page']}_c{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk['text'])
            vectors.append(embedding)
            metadatas.append({'source': chunk['source'], 'page': str(chunk['page']), 'file_type': chunk['file_type'], 'bucket': chunk['bucket'], 'chunk_index': chunk['chunk_index'], 'document_hash': chunk.get('document_hash', '')})
        self._collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
        logger.info(f'Stored {len(ids)} chunk(s). Collection now holds {self._collection.count()} total chunk(s).')
        return len(ids)

    def search(self, query_embedding: list[float], top_k: int=5, bucket: str | None=None) -> list[dict]:
        where_filter = {'bucket': bucket} if bucket else None
        logger.info(f"Searching ChromaDB: top_k={top_k}, bucket={bucket or 'ALL'}, total docs in collection={self._collection.count()}")
        results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_filter, include=['documents', 'metadatas', 'distances'])
        texts = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        retrieved = []
        for text, meta, dist in zip(texts, metadatas, distances):
            retrieved.append({'text': text, 'source': meta.get('source'), 'page': meta.get('page'), 'file_type': meta.get('file_type'), 'bucket': meta.get('bucket'), 'chunk_index': meta.get('chunk_index'), 'distance': round(dist, 4)})
        logger.info(f'Retrieved {len(retrieved)} chunk(s).')
        return retrieved

    def inspect_chunks(self, limit: int=20, offset: int=0, bucket: str | None=None, source: str | None=None, include_embeddings: bool=False) -> dict:
        where = None
        if bucket and source:
            where = {'$and': [{'bucket': bucket}, {'source': source}]}
        elif bucket:
            where = {'bucket': bucket}
        elif source:
            where = {'source': source}
        include = ['documents', 'metadatas']
        if include_embeddings:
            include.append('embeddings')
        result = self._collection.get(where=where, limit=limit, offset=offset, include=include)
        embeddings = result.get('embeddings') if include_embeddings else None
        chunks = []
        for index, chunk_id in enumerate(result.get('ids', [])):
            item = {'id': chunk_id, 'text': result['documents'][index], 'metadata': result['metadatas'][index]}
            if embeddings is not None:
                item['embedding'] = embeddings[index]
            chunks.append(item)
        return {'total_chunks': self._collection.count(), 'offset': offset, 'limit': limit, 'returned_chunks': len(chunks), 'chunks': chunks}

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def document_id(bucket: str, source: str) -> str:
        value = f'{bucket}:{source}'.encode('utf-8')
        return hashlib.sha256(value).hexdigest()[:32]

    def source_fingerprints(self, bucket: str) -> dict[str, str]:
        result = self._collection.get(where={'bucket': bucket}, include=['metadatas'])
        fingerprints = {}
        for metadata in result.get('metadatas', []):
            source = metadata.get('source')
            if source and metadata.get('document_hash'):
                fingerprints[source] = metadata['document_hash']
        return fingerprints

    def indexed_sources(self, bucket: str) -> set[str]:
        result = self._collection.get(where={'bucket': bucket}, include=['metadatas'])
        return {metadata.get('source') for metadata in result.get('metadatas', []) if metadata.get('source')}

    def delete_document(self, bucket: str, source: str) -> None:
        self.delete_sources(bucket, [source])

    def delete_bucket(self, bucket: str) -> None:
        self._collection.delete(where={'bucket': bucket})
        logger.info(f'Removed existing chunks for {bucket}.')

    def delete_sources(self, bucket: str, sources: list[str]) -> None:
        if sources:
            self._collection.delete(where={'$and': [{'bucket': bucket}, {'source': {'$in': sources}}]})
            logger.info(f'Removed existing chunks for {len(sources)} source file(s) in {bucket}.')

    def clear(self) -> None:
        logger.warning('Clearing entire ChromaDB collection!')
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})
        logger.info(f"Collection '{collection_name}' cleared and recreated.")

# Code update
