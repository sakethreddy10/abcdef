from abc import ABC, abstractmethod

class BaseParser(ABC):

    @abstractmethod
    def parse(self, file_path: str, bucket: str) -> list[dict]:
        raise NotImplementedError
