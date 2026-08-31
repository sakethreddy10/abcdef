import logging
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
logger = logging.getLogger(__name__)

def split_text(text: str, chunk_size: int=CHUNK_SIZE, overlap: int=CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_documents(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        text = doc['text']
        source = doc['source']
        text_chunks = split_text(text)
        logger.info(f"  '{source}' -> {len(text_chunks)} chunk(s)")
        for index, chunk_text in enumerate(text_chunks):
            all_chunks.append({'text': chunk_text, 'source': source, 'page': doc['page'], 'file_type': doc['file_type'], 'bucket': doc['bucket'], 'chunk_index': index})
    logger.info(f'Total chunks produced: {len(all_chunks)}')
    return all_chunks
