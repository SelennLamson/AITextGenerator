from src.flexible_models.flexible_bert_embed import FlexibleBERTEmbed
from src.model_evaluation.metrics.flexible_metrics import Metrics

import numpy as np
import pandas as pd

class BertSimilarity(Metrics):
	"""
	Compute the Bert Similarity between pred_P2 and true_P2
		ie : cosine similarity between the embeddings of the two paragraphs
	"""
	def __init__(self, **kwargs):
		"""
		Initialized the BERT model
		:param batch_size: [int] batch size to used for bert
		"""
		super().__init__()
		self.bert_model = FlexibleBERTEmbed(2000, kwargs['batch_size'])

	def __call__(self, predicted_sentences, original_contexts, summarizer):
		"""
        :param predicted_sentences: list[str] batch of sentences corresponding to the generated P2
        :param original_contexts: list[TrainInput] corresponding to original training examples
        :param summarizer: name of the summarizer we use for text generation, from ['PYSUM', 'T5', 'BART', 'KW']
		:return: pd.DataFrame['similarity']
		"""
		# Change notation to match with Thomas old codes
		arr_output = predicted_sentences
		arr_input = [original_context.P2 for original_context in original_contexts]

		model_input = arr_input + arr_output
		vecs = self.bert_model.predict(model_input)
		norms = np.linalg.norm(vecs, axis=1)

		in_vecs = vecs[:len(arr_input)]
		out_vecs = vecs[len(arr_input):]

		in_norms = norms[:len(arr_input)]
		out_norms = norms[len(arr_input):]

		cosine_similarities = np.sum(in_vecs * out_vecs, axis=1) / in_norms / out_norms

		return pd.DataFrame(columns=['similarity'], data=cosine_similarities)
