import torch
from typing import Dict, Optional
from abc import abstractmethod


class ModelHandler:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.classes = None
        self.results: Dict = {}
        self.NUM_OF_WORKER = None

    @abstractmethod
    def process(self,
                input_shape: Optional[int] = 3,
                hidden_units: Optional[int] = 10,
                epochs: int = 5, ):
        raise NotImplemented
