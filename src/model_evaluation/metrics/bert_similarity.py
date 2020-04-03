from src.flexible_models.flexible_bert_embed import FlexibleBERTEmbed
import numpy as np

def bert_similarity(input_p2, output_p2, ner_model: FlexibleBERTEmbed):
	"""
	:param input_p2: the input paragraph, as a string, or an array of strings.
	:param output_p2: the generated paragraph, as a string, or an array of strings of same length as input.
	:param ner_model: FlexibleBERTEmbed model to embed the two paragraphs
	:return cosine similarity between the embeddings of the two paragraphs, or array of scores if multiple paragraphs.
	"""

	arr_input = input_p2 if not isinstance(input_p2, str) else [input_p2]
	arr_output = output_p2 if not isinstance(output_p2, str) else [output_p2]

	model_input = arr_input + arr_output
	vecs = ner_model.predict(model_input)
	norms = np.linalg.norm(vecs, axis=1)

	in_vecs = vecs[:len(arr_input)]
	out_vecs = vecs[len(arr_input):]

	in_norms = norms[:len(arr_input)]
	out_norms = norms[len(arr_input):]

	cosine_similarities = np.sum(in_vecs * out_vecs, axis=1) / in_norms / out_norms

	return cosine_similarities
