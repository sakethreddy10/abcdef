import logging
import ollama
from app.config import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MAX_TOKENS, OLLAMA_MODEL
logger = logging.getLogger(__name__)

class LLMService:

    def __init__(self, base_url: str=OLLAMA_BASE_URL, model: str=OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=self.base_url)
        logger.info(f"LLMService initialized with model '{self.model}' at '{self.base_url}'")

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.warning(f'Ollama server not reachable at {self.base_url}: {e}')
            return False

    def list_available_models(self) -> list[str]:
        try:
            response = self.client.list()
            models = []
            for m in response.get('models', []):
                name = m.get('name') or m.get('model')
                if name:
                    models.append(name)
            return models
        except Exception as e:
            logger.error(f'Failed to list Ollama models: {e}')
            return []

    def generate(self, prompt: str, system_prompt: str | None=None, temperature: float=0.25) -> str:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        logger.info(f"Sending prompt to Ollama model '{self.model}' (temp: {temperature})...")
        try:
            response = self.client.chat(model=self.model, messages=messages, options={'temperature': temperature, 'num_predict': OLLAMA_MAX_TOKENS}, keep_alive=OLLAMA_KEEP_ALIVE)
            answer = response['message']['content']
            return answer.strip()
        except ollama.ResponseError as e:
            if 'not found' in str(e).lower():
                raise RuntimeError(f"Model '{self.model}' was not found in Ollama. Please pull it first by running: ollama pull {self.model}") from e
            raise RuntimeError(f'Ollama generation error: {e}') from e
        except Exception as e:
            raise ConnectionError(f"Could not connect to Ollama at '{self.base_url}'. Ensure Ollama is running (`ollama serve` or open the Ollama app). Error: {e}") from e

    def stream_generate(self, prompt: str, system_prompt: str | None=None, temperature: float=0.2):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        try:
            response_stream = self.client.chat(model=self.model, messages=messages, options={'temperature': temperature, 'num_predict': OLLAMA_MAX_TOKENS}, keep_alive=OLLAMA_KEEP_ALIVE, stream=True)
            for response in response_stream:
                content = response.get('message', {}).get('content', '')
                if content:
                    yield content
        except ollama.ResponseError as e:
            if 'not found' in str(e).lower():
                raise RuntimeError(f"Model '{self.model}' was not found in Ollama. Please pull it first by running: ollama pull {self.model}") from e
            raise RuntimeError(f'Ollama generation error: {e}') from e
        except Exception as e:
            raise ConnectionError(f"Could not connect to Ollama at '{self.base_url}'. Ensure Ollama is running (`ollama serve` or open the Ollama app). Error: {e}") from e

# Code update
