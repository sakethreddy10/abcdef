import logging
import os
import hashlib
from app.parsers.pdf_parser import PDFParser
from app.parsers.docx_parser import DOCXParser
from app.parsers.text_parser import TextParser
from app.parsers.csv_parser import CSVParser
from app.parsers.excel_parser import ExcelParser
from app.parsers.image_parser import ImageParser
logger = logging.getLogger(__name__)
PARSER_MAP = {'.pdf': PDFParser(), '.docx': DOCXParser(), '.txt': TextParser(), '.csv': CSVParser(), '.xlsx': ExcelParser(), '.xls': ExcelParser(), '.png': ImageParser(), '.jpg': ImageParser(), '.jpeg': ImageParser()}

def ingest_file(file_path: str, bucket: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension not in PARSER_MAP:
        supported = ', '.join(PARSER_MAP.keys())
        raise ValueError(f"Unsupported file type: '{extension}'. Supported types: {supported}")
    parser = PARSER_MAP[extension]
    logger.info(f"Ingesting '{os.path.basename(file_path)}' (type: {extension}, bucket: {bucket})")
    documents = parser.parse(file_path, bucket)
    return documents

def ingest_directory(directory_path: str, bucket: str) -> list[dict]:
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f'Not a directory: {directory_path}')
    all_documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isdir(file_path):
            continue
        _, ext = os.path.splitext(filename)
        if ext.lower() not in PARSER_MAP:
            logger.debug(f'Skipping unsupported file: {filename}')
            continue
        try:
            documents = ingest_file(file_path, bucket)
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Failed to ingest '{filename}': {e}")
    logger.info(f'Directory ingestion complete: {directory_path} → {len(all_documents)} document chunk(s) produced')
    return all_documents

def file_fingerprint(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def ingest_directory_incremental(directory_path: str, bucket: str, vector_store):
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f'Not a directory: {directory_path}')
    known_hashes = vector_store.source_fingerprints(bucket)
    indexed_sources = vector_store.indexed_sources(bucket)
    current_sources = set()
    changed_sources = []
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        extension = os.path.splitext(filename)[1].lower()
        if os.path.isdir(file_path) or extension not in PARSER_MAP:
            continue
        current_sources.add(filename)
        fingerprint = file_fingerprint(file_path)
        if known_hashes.get(filename) == fingerprint:
            logger.info(f'Skipping unchanged file: {filename}')
            continue
        try:
            parsed = ingest_file(file_path, bucket)
            for document in parsed:
                document['document_hash'] = fingerprint
            documents.extend(parsed)
            changed_sources.append(filename)
        except Exception as e:
            logger.error(f"Failed to ingest '{filename}': {e}")
    removed_sources = indexed_sources - current_sources
    return (documents, changed_sources, removed_sources)
