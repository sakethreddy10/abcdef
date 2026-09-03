import logging
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.config import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a precise document question-answering assistant.
Answer the user's question using only the supplied Context.

Rules:
1. Identify the context passage that directly answers the question and ignore unrelated passages.
2. Give the answer immediately in one or two concise sentences. Include only the facts needed to answer the question.
3. If the context does not directly support an answer, say: "The provided documents do not contain sufficient information to answer this question."
4. Never guess, infer missing facts, combine unrelated passages, or use outside knowledge.
5. Cite the supporting source at the end of the answer as (Source: filename, page N). If no page exists, use (Source: filename).
6. Do not repeat the question, mention the context or retrieval process, use headings, or add unnecessary explanation.
7. Try to give the answer in detailed Organized way.
"""


class RAGService:
    def __init__(self, retrieval_service: RetrievalService, llm_service: LLMService):
        self.retriever = retrieval_service
        self.llm = llm_service

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        """
        Format retrieved chunks and user question into the final LLM prompt.

        Args:
            question: The user query.
            chunks: Retrieved context chunks with text and metadata.

        Returns:
            str: Formatted prompt for the LLM.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "N/A")
            bucket = chunk.get("bucket", "N/A")
            text = chunk.get("text", "").strip()

            context_parts.append(
                f"[Source: {source} | Page: {page} | Bucket: {bucket}]\n{text}"
            )

        context_str = "\n\n".join(context_parts)

        prompt = (
            f"--- START OF CONTEXT ---\n"
            f"{context_str}\n"
            f"--- END OF CONTEXT ---\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    def query(
        self,
        question: str,
        bucket: str | None = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> dict:

        logger.info(f"RAG query received: '{question}' (bucket: {bucket or 'ALL'})")

        # Step 1: Retrieve relevant context chunks
        chunks = self.retriever.retrieve(question, top_k=top_k, bucket=bucket)

        if not chunks:
            logger.warning("No relevant chunks retrieved from vector store.")
            return {
                "question": question,
                "answer": "No relevant documents found in the selected repository bucket.",
                "bucket": bucket,
                "sources": [],
                "retrieved_chunks": [],
            }

        # Step 2: Build the context-rich prompt
        prompt = self.build_prompt(question, chunks)

        # Step 3: Query Ollama LLM
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=0.2,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise e

        # Step 4: Extract and deduplicate source citations
        unique_sources = []
        seen = set()
        for c in chunks:
            key = (c.get("source"), c.get("page"), c.get("bucket"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(
                    {
                        "document": c.get("source"),
                        "page": c.get("page"),
                        "bucket": c.get("bucket"),
                        "file_type": c.get("file_type"),
                    }
                )

        return {
            "question": question,
            "answer": answer,
            "bucket": bucket,
            "sources": unique_sources,
            "retrieved_chunks": chunks, 
        }

    def stream_query(self, question: str, bucket: str | None = None, top_k: int = RETRIEVAL_TOP_K):
        """Yield retrieval metadata, answer tokens, and final citations."""
        chunks = self.retriever.retrieve(question, top_k=top_k, bucket=bucket)
        if not chunks:
            yield {
                "type": "done",
                "answer": "No relevant documents found in the selected repository bucket.",
                "sources": [],
            }
            return

        sources = self._unique_sources(chunks)
        yield {"type": "sources", "sources": sources}

        prompt = self.build_prompt(question, chunks)
        answer_parts = []
        for token in self.llm.stream_generate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.2,
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}

        yield {"type": "done", "answer": "".join(answer_parts).strip(), "sources": sources}

    @staticmethod
    def _unique_sources(chunks: list[dict]) -> list[dict]:
        unique_sources = []
        seen = set()
        for chunk in chunks:
            key = (chunk.get("source"), chunk.get("page"), chunk.get("bucket"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(
                    {
                        "document": chunk.get("source"),
                        "page": chunk.get("page"),
                        "bucket": chunk.get("bucket"),
                        "file_type": chunk.get("file_type"),
                    }
                )
        return unique_sources
