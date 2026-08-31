import logging
import os
from docx import Document
from app.parsers.base_parser import BaseParser
logger = logging.getLogger(__name__)

class DOCXParser(BaseParser):

    def parse(self, file_path: str, bucket: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'DOCX file not found: {file_path}')
        source_name = os.path.basename(file_path)
        logger.info(f'Parsing DOCX: {source_name}')
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = '\n'.join(paragraphs)
        if not full_text.strip():
            logger.warning(f'No text found in DOCX: {source_name}')
            return []
        logger.info(f'  → Extracted {len(paragraphs)} paragraph(s) from {source_name}')
        return [{'text': full_text.strip(), 'source': source_name, 'page': None, 'file_type': 'docx', 'bucket': bucket}]

def parse_docx(file_path: str, bucket: str) -> list[dict]:
    return DOCXParser().parse(file_path, bucket)
