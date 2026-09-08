import logging
import re
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
logger = logging.getLogger(__name__)

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        limit = min(start + chunk_size, len(text))
        end = limit
        if limit < len(text):
            boundary_pattern = r"\n|[.!?](?=\s)"
            boundaries = [match.end() for match in re.finditer(boundary_pattern, text[start:limit])]
            if boundaries and boundaries[-1] >= chunk_size // 2:
                end = start + boundaries[-1]
            else:
                next_boundary = re.search(boundary_pattern, text[limit:])
                if next_boundary:
                    end = limit + next_boundary.end()

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def split_csv_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0]
    chunks = []
    current_rows = []
    current_length = len(header)
    for row in lines[1:]:
        if current_rows and current_length + len(row) + 1 > chunk_size:
            chunks.append("\n".join([header, *current_rows]))
            current_rows = []
            current_length = len(header)
        current_rows.append(row)
        current_length += len(row) + 1

    if current_rows:
        chunks.append("\n".join([header, *current_rows]))
    return chunks

def chunk_documents(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        text = doc['text']
        source = doc['source']
        text_chunks = split_csv_text(text) if doc['file_type'] == 'csv' else split_text(text)
        logger.info(f"  '{source}' -> {len(text_chunks)} chunk(s)")
        for index, chunk_text in enumerate(text_chunks):
            all_chunks.append({'text': chunk_text, 'source': source, 'page': doc['page'], 'file_type': doc['file_type'], 'bucket': doc['bucket'], 'chunk_index': index})
    logger.info(f'Total chunks produced: {len(all_chunks)}')
    return all_chunks

# Code update
