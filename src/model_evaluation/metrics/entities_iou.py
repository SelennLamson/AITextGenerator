from src.flexible_models.flexible_bert_ner import FlexibleBERTNER

def entities_iou(input_entities, output_p2, ner_model: FlexibleBERTNER):
	"""Evaluates the intersection-over-union of entities in input and output P2.
	:param input_entities: the list of entities inside input P2.
	:param output_p2: the generated paragraph, as a string.
	:param ner_model: FlexibleBERTNER model to detect entities in output P2
	:return intersection-over-union score on required and detected entities
	"""

	output_entities = set()
	model_output = ner_model.predict(output_p2, verbose=0)

	for entities in model_output:
		for ent in entities:
			output_entities.add(ent)

	union = output_entities.union(set(input_entities))
	intersection = output_entities.intersection(set(input_entities))

	return len(union) / len(intersection)
