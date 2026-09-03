import logging
import hashlib

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_PATH, CHROMA_COLLECTION

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Manages all interactions with the ChromaDB vector database.

    Responsibilities:
    - Create / connect to the on-disk ChromaDB collection.
    - Add chunks (text + embedding + metadata) to the collection.
    - Search for chunks similar to a query embedding, with optional bucket filter.
    - Count and clear documents (useful for debugging and testing).

    We initialise the ChromaDB client once in __init__ and reuse it,
    just like we do with the embedding model.
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PATH,
        collection_name: str = CHROMA_COLLECTION,
    ):
        """
        Connect to (or create) the ChromaDB collection on disk.

        Args:
            persist_directory: Folder where ChromaDB saves its data files.
                               Created automatically if it doesn't exist.
            collection_name:   Name of the collection inside ChromaDB.
        """
        logger.info(
            f"Connecting to ChromaDB at '{persist_directory}' "
            f"(collection: '{collection_name}')"
        )

        # PersistentClient stores everything to disk.
        # This means your embeddings survive between Python runs.
        # If you use chromadb.Client() instead, data is lost on exit.
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),  # Disable usage tracking
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity (not L2)
        )

        doc_count = self._collection.count()
        logger.info(
            f"Collection '{collection_name}' ready. "
            f"Currently holds {doc_count} chunk(s)."
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """
        Store chunks (text + metadata) and their pre-computed embeddings.

        Each chunk gets a unique ID so ChromaDB can update/delete it later.
        The ID format is: "<source>_p<page>_c<chunk_index>"

        Args:
            chunks:     List of chunk dicts from ChunkingService.
                        Each must have: text, source, page, file_type, bucket, chunk_index
            embeddings: List of float-lists (one per chunk), from EmbeddingService.
                        Must be the same length as chunks.

        Returns:
            Number of chunks successfully added.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length."
            )

        if not chunks:
            logger.warning("add_chunks() called with empty list — nothing stored.")
            return 0

        # ── Build the four parallel lists ChromaDB expects ───────────────────
        ids        = []
        documents  = []   # The raw text strings
        metadatas  = []   # Dicts of metadata (bucket, source, page, etc.)
        vectors    = []   # The float-lists from EmbeddingService

        for chunk, embedding in zip(chunks, embeddings):
            # Build a unique, human-readable ID for each chunk.
            # Using source + page + chunk_index makes it deterministic —
            # you can check if a chunk is already stored without querying.
            chunk_id = (
                f"{self.document_id(chunk['bucket'], chunk['source'])}"
                f"_p{chunk['page']}"
                f"_c{chunk['chunk_index']}"
            )

            ids.append(chunk_id)
            documents.append(chunk["text"])
            vectors.append(embedding)

            # ChromaDB metadata values must be str, int, float, or bool.
            # None is NOT allowed — convert it to "None" string.
            metadatas.append(
                {
                    "source":      chunk["source"],
                    "page":        str(chunk["page"]),        # str to handle None
                    "file_type":   chunk["file_type"],
                    "bucket":      chunk["bucket"],
                    "chunk_index": chunk["chunk_index"],
                    "document_hash": chunk.get("document_hash", ""),
                }
            )

        # collection.add() stores everything in one batch call.
        # ChromaDB builds its HNSW index internally.
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

        logger.info(
            f"Stored {len(ids)} chunk(s). "
            f"Collection now holds {self._collection.count()} total chunk(s)."
        )
        return len(ids)

    # ── Read / Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        bucket: str | None = None,
    ) -> list[dict]:
        """
        Find the top-K chunks most similar to the query embedding.

        WHAT "SIMILAR" MEANS HERE:
            ChromaDB uses cosine similarity (we configured "hnsw:space": "cosine").
            Cosine similarity measures the ANGLE between two vectors — the smaller
            the angle, the more similar the meaning.
            Score of 1.0 = identical direction (perfect match).
            Score of 0.0 = no relationship.

        BUCKET FILTERING:
            If bucket is provided (e.g. "bucket_1"), ChromaDB only searches
            chunks where metadata["bucket"] == "bucket_1".
            This is how we isolate the two document sets cleanly.

        Args:
            query_embedding: A single float-list (the embedded question).
            top_k:           How many chunks to return (default: 5).
            bucket:          Optional. If given, restrict search to that bucket.

        Returns:
            List of dicts, each representing one retrieved chunk:
            {
                "text":        "the chunk text",
                "source":      "document.pdf",
                "page":        "3",
                "file_type":   "pdf",
                "bucket":      "bucket_1",
                "chunk_index": 2,
                "distance":    0.21   ← lower = more similar (cosine distance)
            }
            Ordered by similarity (most similar first).
        """
        # Build the where-filter for bucket isolation.
        # where=None means "search all chunks regardless of bucket".
        where_filter = {"bucket": bucket} if bucket else None

        logger.info(
            f"Searching ChromaDB: top_k={top_k}, "
            f"bucket={bucket or 'ALL'}, "
            f"total docs in collection={self._collection.count()}"
        )

        results = self._collection.query(
            query_embeddings=[query_embedding],  # Must be a list-of-lists
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        texts     = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Flatten into a clean list of dicts for the caller.
        retrieved = []
        for text, meta, dist in zip(texts, metadatas, distances):
            retrieved.append(
                {
                    "text":        text,
                    "source":      meta.get("source"),
                    "page":        meta.get("page"),
                    "file_type":   meta.get("file_type"),
                    "bucket":      meta.get("bucket"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance":    round(dist, 4),   # Cosine distance (lower = better)
                }
            )

        logger.info(f"Retrieved {len(retrieved)} chunk(s).")
        return retrieved

    def inspect_chunks(
        self,
        limit: int = 20,
        offset: int = 0,
        bucket: str | None = None,
        source: str | None = None,
        include_embeddings: bool = False,
    ) -> dict:
        """Return stored chunks for inspection, optionally filtered by source."""
        where = None
        if bucket and source:
            where = {"$and": [{"bucket": bucket}, {"source": source}]}
        elif bucket:
            where = {"bucket": bucket}
        elif source:
            where = {"source": source}

        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        result = self._collection.get(
            where=where,
            limit=limit,
            offset=offset,
            include=include,
        )
        embeddings = result.get("embeddings") if include_embeddings else None
        chunks = []
        for index, chunk_id in enumerate(result.get("ids", [])):
            item = {
                "id": chunk_id,
                "text": result["documents"][index],
                "metadata": result["metadatas"][index],
            }
            if embeddings is not None:
                item["embedding"] = embeddings[index]
            chunks.append(item)

        return {
            "total_chunks": self._collection.count(),
            "offset": offset,
            "limit": limit,
            "returned_chunks": len(chunks),
            "chunks": chunks,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of chunks currently in the collection."""
        return self._collection.count()

    @staticmethod
    def document_id(bucket: str, source: str) -> str:
        value = f"{bucket}:{source}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()[:32]

    def source_fingerprints(self, bucket: str) -> dict[str, str]:
        result = self._collection.get(
            where={"bucket": bucket},
            include=["metadatas"],
        )
        fingerprints = {}
        for metadata in result.get("metadatas", []):
            source = metadata.get("source")
            if source and metadata.get("document_hash"):
                fingerprints[source] = metadata["document_hash"]
        return fingerprints

    def indexed_sources(self, bucket: str) -> set[str]:
        result = self._collection.get(
            where={"bucket": bucket},
            include=["metadatas"],
        )
        return {
            metadata.get("source")
            for metadata in result.get("metadatas", [])
            if metadata.get("source")
        }

    def delete_document(self, bucket: str, source: str) -> None:
        self.delete_sources(bucket, [source])

    def delete_bucket(self, bucket: str) -> None:
        """Remove all indexed chunks for one bucket before a full re-index."""
        self._collection.delete(where={"bucket": bucket})
        logger.info(f"Removed existing chunks for {bucket}.")

    def delete_sources(self, bucket: str, sources: list[str]) -> None:
        """Remove existing chunks for selected files before replacing uploads."""
        if sources:
            self._collection.delete(
                where={"$and": [{"bucket": bucket}, {"source": {"$in": sources}}]}
            )
            logger.info(f"Removed existing chunks for {len(sources)} source file(s) in {bucket}.")

    def clear(self) -> None:
        """
        Delete ALL chunks from the collection.

        USE WITH CARE — this removes everything in the collection.
        Useful when re-ingesting documents from scratch during development.
        """
        logger.warning("Clearing entire ChromaDB collection!")
        # Delete the collection and recreate it fresh.
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{collection_name}' cleared and recreated.")
