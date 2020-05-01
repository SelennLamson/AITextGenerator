from typing import List
import pandas as pd
from abc import ABC
from src.torch_loader.vectorize_input import TrainInput


class Metrics(ABC):
    """
    Abstract class for which all custom metrics must inherated
    """

    def __init__(self, **kwargs):
        """
        Will be use to initialize transformer model if needed
        """
        pass

    def __call__(self, predicted_sentences: List[str], original_contexts: List[TrainInput]) -> pd.DataFrame:
        """
        :param predicted_sentences: list[str] batch of sentences corresponding to the generated P2
        :param original_contexts: list[TrainInput] corresponding to original training examples
        :return: pd.DataFrame containing the metrics results, one row for each predicted sentence
        """
        pass
