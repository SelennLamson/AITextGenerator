from .flexible_model import FlexibleModel
from src.third_party.BERT_NER.bert import Ner
from typing import List, Dict, Tuple
from src.utils import *


class FlexibleBERTNER(FlexibleModel):
	def __init__(self, bert_path: str, batch_size:int, max_length: int):
		"""
		Initializes a BERT-NER model.
		:param bert_path: Path to BERT-NER weights
		:param batch_size: Batch-size to use when predicting.
		:param max_length: The maximum length (in char) the model can handle
		"""
		super().__init__()
		self.bert_model = Ner(bert_path)
		self.batch_size = batch_size
		self.max_length = max_length

	def predict(self, inputs: List[str], verbose: int = 1) -> List[Dict[str, Tuple[str, float]]]:
		"""
		Performs NER on strings of any length.
		:param inputs: list of strings.
		:param verbose: 1 to display progress, 0 for silent execution.
		:return: List[Dict[string: entities, Tuple[string: type, float: confidence]]]
		"""

		zipped_tokens = [list(zip(*self.bert_model.tokenize(inp + '.'))) for inp in inputs]

		split_tokens, split_information = token_batch_splitter(zipped_tokens, self.max_length - 2)

		outputs = []
		start_i = 0
		while start_i < len(split_tokens):
			if verbose >= 1:
				print("\rNER - {:.2f}%".format(start_i / len(split_tokens) * 100), end="")

			input_tokens = [[s[0] for s in st] for st in split_tokens[start_i:start_i + self.batch_size]]
			input_valid_positions = [[s[1] for s in st] for st in split_tokens[start_i:start_i + self.batch_size]]


			# inputs = [s + '.' for s in split_strings[start_i:start_i + self.batch_size]]
			outputs += self.bert_model.predict_batch(input_tokens, input_valid_positions)
			start_i += self.batch_size

		return batch_merger(outputs, split_information, merge_function=self.merge_entities, apply_on_single=True)

	def merge_entities(self, outputs):
		# Merging predictions together, using probability rule: p(A or B) = p(A) + p(B) - p(A)*p(B)
		entities = dict()
		current_entity = None
		current_confidence = 0
		current_tag = None
		for out in outputs:
			for o in out:
				tag = o['tag'][2:]
				begin = o['tag'][0] == 'B'
				entity = o['word'].replace('.', '').replace(' ', '')
				confidence = o['confidence']

				# 1. If we encounter a new entity, but current one is not registered yet
				# OR
				# 2. We see no tag anymore but we had an entity in mind, so we register it
				if (tag != "" and begin and current_entity is not None) or \
						(tag == "" and current_entity is not None):
					if current_entity in entities:
						# We already encountered this entity in a previous sequence
						prev_tag, prev_conf = entities[current_entity]
						if prev_tag == current_tag:
							# This is the same tag, we apply p(A or B) rule
							conf = prev_conf + current_confidence - prev_conf * current_confidence
							entities[current_entity] = (prev_tag, conf)
						elif prev_conf < current_confidence:
							# This is not the same tag as before, we just keep the best one
							entities[current_entity] = (current_tag, current_confidence)
					else:
						# This is the first time we encounter this entity
						entities[current_entity] = (current_tag, current_confidence)

					# After registering, we reset the entity to None
					current_entity = None

				# Now, we process the current tag
				if tag != "":  # We have a Named Entity
					if begin:  # It is a new one
						current_entity = entity
						current_confidence = confidence
						current_tag = tag
					elif current_entity is not None and not begin:  # It is continuing the current one
						current_entity += " " + entity
						current_confidence = current_confidence * 0.7 + confidence * 0.3  # Simple heuristic to merge confidences

		return entities
