try:
    import pymupdf as fitz
except ImportError:
    import fitz
import logging
import os
from PIL import Image
import pytesseract
from app.config import TESSERACT_CMD
from app.parsers.base_parser import BaseParser
logger = logging.getLogger(__name__)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
MIN_TEXT_LENGTH = 20

def _ocr_page(page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)
    return pytesseract.image_to_string(image, lang='eng').strip()

class PDFParser(BaseParser):

    def parse(self, file_path: str, bucket: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'PDF file not found: {file_path}')
        source_name = os.path.basename(file_path)
        pages = []
        logger.info(f'Parsing PDF: {source_name}')
        with fitz.open(file_path) as pdf_document:
            total_pages = len(pdf_document)
            logger.info(f'  → {total_pages} page(s) found in {source_name}')
            for page_index in range(total_pages):
                page_number = page_index + 1
                page = pdf_document.load_page(page_index)
                raw_text = page.get_text()
                cleaned_text = raw_text.strip()
                if len(cleaned_text) < MIN_TEXT_LENGTH:
                    logger.info(f'  → Page {page_number} has insufficient PDF text; trying OCR.')
                    try:
                        ocr_text = _ocr_page(page)
                    except pytesseract.TesseractNotFoundError:
                        logger.error('Tesseract is not installed or not found on PATH. Please install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki')
                        raise
                    if ocr_text:
                        cleaned_text = ocr_text
                if not cleaned_text:
                    logger.warning(f"  → Page {page_number} of '{source_name}' has no readable text. Skipping.")
                    continue
                pages.append({'text': cleaned_text, 'source': source_name, 'page': page_number, 'file_type': 'pdf', 'bucket': bucket})
        if not pages:
            logger.warning(f"No text could be extracted from '{source_name}'. The PDF may be entirely image-based.")
        logger.info(f'  → Extracted {len(pages)} page(s) with text from {source_name}')
        return pages

def parse_pdf(file_path: str, bucket: str) -> list[dict]:
    return PDFParser().parse(file_path, bucket)
