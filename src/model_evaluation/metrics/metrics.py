from typing import List
import pandas as pd
from abc import ABC
from src.torch_loader.vectorize_input import TrainInput

class Metrics(ABC):
    def __init__(self):
        pass

    def __call__(self, predicted_sentences:List[str], original_contexts:List[TrainInput]) -> pd.DataFrame:
        pass