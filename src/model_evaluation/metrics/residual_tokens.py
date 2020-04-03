from typing import List
import numpy as np

def residual_tokens(pred_paragraphs: List[str]) -> List[float]:
	"""Counts how many special tokens are in the generated paragraphs, where it shouldn't be.
	:param pred_paragraphs: list of generated paragraph as strings.
	:return List[float]
	"""
	return np.array([(pred.count('[') + pred.count(']')) / 2 for pred in pred_paragraphs])
