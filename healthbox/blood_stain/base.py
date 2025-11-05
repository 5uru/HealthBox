from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from PIL import Image

class Model(ABC):

    @abstractmethod
    def predict(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        # Placeholder for prediction logic
        pass