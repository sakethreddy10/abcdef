import os
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")
if not TESSERACT_CMD:
	local_tesseract = os.path.join(
		os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"
	)
	if os.path.exists(local_tesseract):
		TESSERACT_CMD = local_tesseract

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "1024"))
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./vectorstore")

CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_documents")

# BGE base provides stronger English retrieval quality than MiniLM.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "450"))

CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))


RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
RETRIEVAL_MAX_DISTANCE: float = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.72"))


# ── Data Paths ────────────────────────────────────────────────────────────────
BUCKET_1_PATH: str = os.getenv("BUCKET_1_PATH", "./data/bucket_1")
BUCKET_2_PATH: str = os.getenv("BUCKET_2_PATH", "./data/bucket_2")
