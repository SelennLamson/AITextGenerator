import numpy as np
import pandas as pd
from gensim.summarization import keywords
from src.model_evaluation.metrics.flexible_metrics import Metrics


class KwIou(Metrics):
	"""
	Study the number of identical keywords in P2 and the generated P2.
	"""

	def __init__(self):
		self.model = keywords

	def __call__(self, predicted_sentences, original_contexts):
		"""
		:param predicted_sentences: generated P2 by GPT2
		:param original_contexts: original P2
		:return: the iou score between list of keywords in original P2 and generated P2
		"""
		kw_in_pred = self.model(predicted_sentences, lemmatize=False, pos_filter=('NN', 'JJ', 'VB')).split('\n')
		kw_in_text = self.model(original_contexts, lemmatize=False, pos_filter=('NN', 'JJ', 'VB')).split('\n')

		if len(kw_in_pred) == 0 and len(kw_in_text) == 0:
			iou = 'NaN'
		else:
			iou = len(set(kw_in_pred).intersection(kw_in_text)) / len(set(kw_in_pred).union(kw_in_text))

		return iou


class KwCount(Metrics):
	"""
	Study the number of identical keywords in P2 and the generated P2.
	"""

	def __init__(self):
		self.model = keywords

	def __call__(self, predicted_sentences, original_contexts):
		"""
		:param predicted_sentences: generated P2 by GPT2
		:param original_contexts: original P2
		:return: proportion of number of original P2 keywords existent in generated P2
		"""
		kw_in_text = self.model(original_contexts, lemmatize=False, pos_filter=('NN', 'JJ', 'VB')).split('\n')

		count = 0
		for kw in kw_in_text:
			if kw in predicted_sentences.lower():
				count += 1
		proportion = count / len(set(kw_in_text))

		if len(set(kw_in_text)) == 0:
			proportion = 'NaN'

		return proportion

