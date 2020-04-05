import json
import os
from typing import List, Dict, Tuple

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

from src.utils import *
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.flexible_models.flexible_bert_embed import FlexibleBERTEmbed
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.model_evaluation.metrics.bert_similarity import bert_similarity
from src.model_evaluation.metrics.entities_iou import entities_iou
from src.model_evaluation.metrics.gpt2_perplexity import gpt2_perplexity
from src.model_evaluation.metrics.residual_tokens import residual_tokens


class GPT2BenchmarkScript:
	def __init__(self, file_ids: List[str], batch_size: int = 1):
		"""Initializes a GPT-2 Benchmark script that will perform text generation on the paragraphs of given files.
		Call the script using parentheses to launch it.
		:param file_ids: list of book ids, the part of the preproc file before "_preproc.json".
		:param batch_size: number of simultaneous text generations.
		"""

		# Filtering file ids on files that really exist in the preproc folder (paragraphs split already)
		files = [f for f in [PREPROC_PATH + fid + PREPROC_SUFFIX for fid in file_ids] if os.path.exists(f)]

		# Retrieving paragraph texts of each files:
		self.paragraphs = []
		self.next_paragraphs = []
		for f in files:
			json_data = json.load(open(f, 'r', encoding='utf-8'))
			book_paragraphs = [p['text'] for p in json_data['paragraphs']]

			self.paragraphs += book_paragraphs[:-1]
			self.next_paragraphs += book_paragraphs[1:]

		self.batch_size = batch_size

	def __call__(self, generations_path:str, results_path:str, compute_bert_similarity=False, compute_entites_iou=False, compute_gpt2_perplexity=False, compute_residual_tokens=False, verbose: int = 1):
		"""Generates texts at generation_path and computes given metrics on them.
		:param generations_path: The path where text generations can be found.
		:param results_path: The path where results should be saved.
		:param compute_bert_similarity: Should "BERT similarity" metric be computed?
		:param compute_entites_iou: Should "Entities I-o-U" metric be computed?
		:param compute_gpt2_perplexity: Should "GPT-2 perplexity" metric be computed?
		:param compute_residual_tokens: Should "Residual tokens" metric be computed?
		:param verbose: 0 for silent execution, 1 for progress.
		"""
		self.generateTexts(generations_path, verbose)
		self.computeMetrics(generations_path, results_path, compute_bert_similarity, compute_entites_iou, compute_gpt2_perplexity, compute_residual_tokens, verbose)

	def generate_texts(self, generations_path:str, verbose: int = 1):
		"""Starts the text generation on all paragraphs.
		:param generations_path: The path where text generations should be saved.
		:param verbose: 0 for silent execution, 1 for progress.
		"""
		if verbose:
			print("Converting paragraphs to tokens...", end="")

		tokenized_paragraphs = [self.tokenizer.encode(' ' + p) for p in self.paragraphs]
		tokenized_nexts = [self.tokenizer.encode(' ' + p) for p in self.next_paragraphs]
		max_length = 1020
		mean_next = int(sum([len(nex) for nex in tokenized_nexts]) / len(tokenized_nexts))

		input_ids = []
		original_nexts = []
		for i, (tok, nex) in enumerate(zip(tokenized_paragraphs, tokenized_nexts)):
			total_length = len(tok) + mean_next
			if total_length > max_length:
				if len(nex) > max_length:
					continue
				else:
					input_ids.append(tok[total_length - max_length:])
					original_nexts.append(self.next_paragraphs[i])
			elif total_length < max_length:
				input_ids.append([self.tokenizer.pad_token_id] * (max_length - total_length) + tok)
				original_nexts.append(self.next_paragraphs[i])
			else:
				input_ids.append(tok)
				original_nexts.append(self.next_paragraphs[i])

		# We don't need non-padded paragraphs anymore
		del tokenized_paragraphs
		del tokenized_nexts

		assert len(input_ids) == len(original_nexts)
		assert all([len(inp) == max_length - mean_next for inp in input_ids])

		if verbose:
			print("\rLoading generative model...", end="")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		decoding_strategy = DEFAULT_DECODING_STRATEGY.copy()
		decoding_strategy['max_length'] = mean_next
		decoding_strategy['min_length'] = int(mean_next * 0.8)
		flexible_model = FlexibleGPT2(model, tokenizer, decoding_strategy)

		if verbose:
			print("\rGenerating texts... 0.00%", end="")

		generations = []
		current_index = 0
		while current_index < len(input_ids):
			if verbose:
				print("\rGenerating texts... {:.2f}%".format(current_index / len(input_ids) * 100), end="")

			current_batch = torch.LongTensor(input_ids[current_index:min(current_index + self.batch_size, len(input_ids))]).to(device)
			generations += flexible_model(current_batch, nb_samples=1)
			current_index += self.batch_size

		# Freeing space
		del flexible_model
		del model
		del tokenizer

		if verbose:
			print("\rSaving generated texts...", end="")

		assert len(generations) == len(original_nexts)
		json_data = [{"generated": gen, "original": ori} for gen, ori in zip(generations, original_nexts)]
		json.dump(json_data, open(generations_path, 'w', encoding='utf-8'))

		if verbose:
			print("\rGeneration successfull.")

	def compute_metrics(self, generations_path:str, results_path:str, compute_bert_similarity=False, compute_entites_iou=False, compute_gpt2_perplexity=False, compute_residual_tokens=False, verbose: int = 1):
		"""Computes the selected metrics on generated texts.
		:param generations_path: The path where text generations can be found.
		:param results_path: The path where results should be saved.
		:param compute_bert_similarity: Should "BERT similarity" metric be computed?
		:param compute_entites_iou: Should "Entities I-o-U" metric be computed?
		:param compute_gpt2_perplexity: Should "GPT-2 perplexity" metric be computed?
		:param compute_residual_tokens: Should "Residual tokens" metric be computed?
		:param verbose: 0 for silent execution, 1 for progress.
		"""

		if verbose:
			print("Computing metrics...", end="")

		generations = json.load(open(generations_path, 'r', encoding='utf-8'))
		generated = [g['generated'] for g in generations]
		originals = [g['original'] for g in generations]

		if os.path.exists(results_path):
			results = json.load(open(results_path, 'r'))
		else:
			results = dict()
			results['per_paragraph'] = [dict() for _ in range(len(originals))]

		per_paragraph = results['per_paragraph']

		def register_stats(_array, _name):
			results[_name] = {'mean': np.mean(_array), 'max': np.max(_array), 'min': np.min(_array), 'median': np.median(_array)}

		if compute_bert_similarity:
			bert_embed_model = FlexibleBERTEmbed(2000, self.batch_size)
			bert_similarities = bert_similarity(originals, generated, bert_embed_model, verbose)

			# Freeing space
			del bert_embed_model

			if verbose:
				print("\rRegistering bert simirality results...", end="")
			for i, sim in enumerate(bert_similarities):
				per_paragraph[i]['bert_similarity'] = sim
			register_stats(bert_similarities, 'bert_similarity')

		if compute_entites_iou:
			bert_ner_model = FlexibleBERTNER(BERT_NER_LARGE, batch_size=self.batch_size)
			ent_ious = entities_iou(originals, generated, bert_ner_model)

			ent_ious = np.sum([ent_ious[key] for key in ENTITY_TAGS], axis=0) / len(ENTITY_TAGS)

			# Freeing space
			del bert_ner_model

			if verbose:
				print("\rRegistering entities I-o-U results...", end="")
			for i, ent in enumerate(ent_ious):
				per_paragraph[i]['entities_iou'] = ent
			register_stats(ent_ious, 'entities_iou')

		if compute_gpt2_perplexity:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
			tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
			flexible_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

			gpt2_gen_perplexities = gpt2_perplexity(generated, flexible_model, device)
			gpt2_ori_perplexities = gpt2_perplexity(originals, flexible_model, device)
			gpt2_perplexities = gpt2_gen_perplexities / gpt2_ori_perplexities

			# Freeing space
			del flexible_model
			del tokenizer
			del model

			if verbose:
				print("\rRegistering GPT-2 perplexities results...", end="")
			for i, perplexity in enumerate(gpt2_perplexities):
				per_paragraph[i]['gpt2_perplexity'] = perplexity
			register_stats(gpt2_perplexities, 'gpt2_perplexity')

		if compute_residual_tokens:
			res_toks = residual_tokens(generated)
			if verbose:
				print("\rRegistering Residual Tokens results...", end="")
			for i, res in enumerate(res_toks):
				per_paragraph[i]['residual_tokens'] = res
			register_stats(res_toks, 'residual_tokens')

		json.dump(results, open(results_path, 'w'))

		if verbose:
			print("\rMetrics computed successfully.")



