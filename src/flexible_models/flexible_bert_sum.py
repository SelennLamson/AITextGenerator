from typing import List
from .flexible_model import FlexibleSummarizer
from src.utils import *
from summarizer import Summarizer


class FlexibleBERTSum(FlexibleSummarizer):
	def __init__(self):
		"""
		Initializes a BERT-SUM model.
		:param min_length: The min length of the summary
		"""
		super().__init__()
		self.bert_sum_model = Summarizer()
		print(self.bert_sum_model)
		# self.min_length = None

	def predict(self, inputs: List[str]) -> List[str]:
		"""
		Performs summarization on each paragraph
		:param inputs: list of strings.
		:return: one summary
		"""
		outputs = self.bert_sum_model(inputs)  # self.min_length
		result = ''.join(outputs)

		return result

