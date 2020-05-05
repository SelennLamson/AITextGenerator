from src.model_evaluation.metrics.flexible_metrics import Metrics

from rouge import Rouge
import pandas as pd
import numpy as np


class RougeScore(Metrics):
	"""
	Compute BleuScore between generated P2 and pred P2
	"""

	def __init__(self, **kwargs):
		super().__init__()
		self.rouge = Rouge()

	def __call__(self, predicted_sentences, original_contexts):
		"""
		:param predicted_sentences: generated P2 by GPT2
		:param original_contexts: original P2
		:return: bleu score for each pair of pred, orig sentence
		"""
		df_results = pd.DataFrame(columns=["rouge_1", "rouge_2", "rouge_l"],
								  data=np.zeros((len(predicted_sentences), 3)))

		for i, (predicted_sentence, original_context) in enumerate(zip(predicted_sentences, original_contexts)):
			try:
				rouge_score = self.rouge.get_scores(predicted_sentence, original_context.P2)
				df_results.loc[i, "rouge_1"] = rouge_score[0]['rouge-1']['f']
				df_results.loc[i, "rouge_2"] = rouge_score[0]['rouge-2']['f']
				df_results.loc[i, "rouge_l"] = rouge_score[0]['rouge-l']['f']
			except ValueError:
				df_results.loc[i, "rouge_1"] = 'NaN'
				df_results.loc[i, "rouge_2"] = 'NaN'
				df_results.loc[i, "rouge_l"] = 'NaN'

		return df_results
