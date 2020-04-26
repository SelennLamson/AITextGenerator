from nltk.translate.bleu_score import sentence_bleu
from src.model_evaluation.metrics.flexible_metrics import Metrics
import pandas as pd
import numpy as np

class BleuScore(Metrics):
	"""
	Compute BleuScore between generated P2 and pred P2
	"""

	def __init__(self, **kwargs):
		super().__init__()
		pass

	def __call__(self, predicted_sentences, original_contexts):
		"""
		:param predicted_sentences: generated P2 by GPT2
		:param original_contexts: original P2
		:return: bleu score for each pair of pred, orig sentence
		"""
		df_results = pd.DataFrame(columns=["bleu_score"], data=np.zeros((len(predicted_sentences),1)))

		for i, (predicted_sentence, original_context) in enumerate(zip(predicted_sentences, original_contexts)):
			try:
				df_results.loc[i, "bleu_score"] = sentence_bleu(original_context.P2, predicted_sentence)
			except ValueError:
				df_results.loc[i, "bleu_score"] = 'NaN'

		return df_results
