import logging
import os
import pandas as pd
from app.parsers.base_parser import BaseParser
logger = logging.getLogger(__name__)

class CSVParser(BaseParser):

    def parse(self, file_path: str, bucket: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'CSV file not found: {file_path}')
        source_name = os.path.basename(file_path)
        logger.info(f'Parsing CSV: {source_name}')
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f'CSV file is empty: {source_name}')
            return []
        row_texts = []
        for _, row in df.iterrows():
            row_text = ' | '.join((f'{col}: {val}' for col, val in row.items()))
            row_texts.append(row_text)
        full_text = '\n'.join(row_texts)
        logger.info(f'  → Converted {len(df)} row(s) from {source_name}')
        return [{'text': full_text, 'source': source_name, 'page': None, 'file_type': 'csv', 'bucket': bucket}]

def parse_csv(file_path: str, bucket: str) -> list[dict]:
    return CSVParser().parse(file_path, bucket)

# Code update
