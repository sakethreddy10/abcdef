import logging
import os
from PIL import Image, ImageOps
import pytesseract
from app.config import TESSERACT_CMD
from app.parsers.base_parser import BaseParser
logger = logging.getLogger(__name__)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

class ImageParser(BaseParser):

    def parse(self, file_path: str, bucket: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Image file not found: {file_path}')
        source_name = os.path.basename(file_path)
        logger.info(f'Parsing image (OCR): {source_name}')
        try:
            with Image.open(file_path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                if max(image.size) < 1600:
                    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
                image = ImageOps.autocontrast(ImageOps.grayscale(image))
                extracted_text = pytesseract.image_to_string(image, lang='eng')
            extracted_text = extracted_text.strip()
            if not extracted_text:
                logger.warning(f'OCR found no text in image: {source_name}. The image may not contain readable text, or quality is too low.')
                return []
            logger.info(f'  → OCR extracted {len(extracted_text)} character(s) from {source_name}')
            return [{'text': extracted_text, 'source': source_name, 'page': None, 'file_type': os.path.splitext(source_name)[1].lstrip('.').lower(), 'bucket': bucket}]
        except pytesseract.TesseractNotFoundError:
            logger.error('Tesseract is not installed or not found on PATH. Please install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki')
            raise

def parse_image(file_path: str, bucket: str) -> list[dict]:
    return ImageParser().parse(file_path, bucket)
