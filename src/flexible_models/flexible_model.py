from typing import List
from abc import ABC

class FlexibleModel(ABC):
	def __init__(self):
		pass

	def predict(self, *args, **kwargs):
		pass

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)


class FlexibleSummarizer(ABC):
	def __init__(self):
		pass

	def predict(self, inputs:List[str]) -> List[str]:
		pass

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)
