# TODO: Import BERT-SUM files
from typing import List

from .flexible_model import FlexibleSummarizer
from src.utils import *


class FlexibleBERTSum(FlexibleSummarizer):
	def __init__(self, bert_path: str, max_length: int):
		"""
		Initializes a BERT-SUM model.
		:param bert_path: Path to BERT-SUM weights
		:param max_length: The maximum length (in char) the model can handle
		"""
		super().__init__()
		self.bert_model = None  # TODO: Load BERT-SUM model
		self.max_length = max_length

	def predict(self, inputs: List[str]) -> List[str]:
		"""
		Performs summarization on strings of any length.
		:param inputs: list of strings.
		:return: list of summaries.
		"""
		split_strings, split_information = text_batch_splitter(inputs, self.max_length)

		outputs = split_strings  # TODO: Apply BERT-SUM model to split_strings

		return batch_merger(outputs, split_information, merge_function=self.merge_summaries)

	def merge_summaries(self, summaries: List[str]) -> str:
		# TODO: Write function to merge summaries together. For now, it's only joining them with whitespaces.
		return ' '.join(summaries)
