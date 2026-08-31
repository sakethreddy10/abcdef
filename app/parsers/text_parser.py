import logging
import os
from app.parsers.base_parser import BaseParser
logger = logging.getLogger(__name__)

class TextParser(BaseParser):

    def parse(self, file_path: str, bucket: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Text file not found: {file_path}')
        source_name = os.path.basename(file_path)
        logger.info(f'Parsing TXT: {source_name}')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f'UTF-8 failed for {source_name}, retrying with latin-1')
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        content = content.strip()
        if not content:
            logger.warning(f'Text file is empty: {source_name}')
            return []
        logger.info(f'  → Read {len(content)} characters from {source_name}')
        return [{'text': content, 'source': source_name, 'page': None, 'file_type': 'txt', 'bucket': bucket}]

def parse_text(file_path: str, bucket: str) -> list[dict]:
    return TextParser().parse(file_path, bucket)

# Code update
