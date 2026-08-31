import logging
import os
import pandas as pd
from app.parsers.base_parser import BaseParser
logger = logging.getLogger(__name__)

class ExcelParser(BaseParser):

    def parse(self, file_path: str, bucket: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Excel file not found: {file_path}')
        source_name = os.path.basename(file_path)
        logger.info(f'Parsing Excel: {source_name}')
        all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        results = []
        for sheet_name, df in all_sheets.items():
            if df.empty:
                logger.warning(f"  → Sheet '{sheet_name}' in {source_name} is empty. Skipping.")
                continue
            row_texts = []
            for _, row in df.iterrows():
                row_text = ' | '.join((f'{col}: {val}' for col, val in row.items()))
                row_texts.append(row_text)
            full_text = '\n'.join(row_texts)
            logger.info(f"  → Sheet '{sheet_name}': {len(df)} row(s)")
            results.append({'text': full_text, 'source': f'{source_name} [Sheet: {sheet_name}]', 'page': None, 'file_type': 'xlsx', 'bucket': bucket})
        if not results:
            logger.warning(f'No data found in any sheet of: {source_name}')
        return results

    def parse_excel(file_path: str, bucket: str) -> list[dict]:
        return ExcelParser().parse(file_path, bucket)

# Code update
