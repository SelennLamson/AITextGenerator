from typing import List
from abc import ABC

class FlexibleModel(ABC):
	def __init__(self):
		pass

	def predict(self, *args, **kwargs):
		pass

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)

class FlexibleSummarizer(FlexibleModel, ABC):
	def predict(self, inputs:List[str]) -> List[str]:
		"""
		Performs summarization on strings of any length.
		:param inputs: list of strings.
		:return: list of summaries.
		"""
		return []
