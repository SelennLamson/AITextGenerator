#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

from typing import List, Any, Tuple
import functools
from math import ceil
from enum import Enum
from collections import namedtuple
import random
import os
import json

# Constants
NOVEL_PATH = 'data/novel/'
NOVEL_SUFFIX = '_novel.json'
PREPROC_PATH = 'data/preproc/'
PREPROC_SUFFIX = '_preproc.json'
METADATA_PATH = 'data/metadata/files/'
METADATA_SUFFIX = '.json'
METADATA_ROOT = 'data/metadata/'

ENTITY_CLASSES = ("persons", "organisations", "locations", "misc")
ENTITY_TAGS = ("PER", "ORG", "LOC", "MISC")

BERT_NER_LARGE = 'models/entity_recognition/BERT_NER_Large/'
BERT_NER_BASE = 'models/entity_recognition/BERT_NER_Base/'

FOLDER_NAME_KW = 'data/Preproc_KW/'
PREFIX_KW = 'KW_'
FOLDER_NAME_T5 = 'data/Preproc_T5/'
PREFIX_T5 = 'T5_'
FOLDER_NAME_BART = 'data/Preproc_BART/'
PREFIX_BART = 'BART_'
FOLDER_NAME_BERTSUM = 'data/Preproc_BERTSUM/'
PREFIX_BERTSUM = 'BERTSUM_'
FOLDER_NAME_PYSUM = 'data/Preproc_PYSUM/'
PREFIX_PYSUM = 'PYSUM_'

GPT2_BLOCK_SIZE = 1020

DEFAULT_DECODING_STRATEGY = {
	'do_sample': True,
	'min_length': 0,
	'max_length': GPT2_BLOCK_SIZE,
	'top_k': 50,
	'top_p': 0.95
}
BART_DECODING_STRAT = \
	{'temperature':1.25,
	 'top_p':0.9,
	 'min_length':25,
	 'max_length':65,
	 'repetition_penalty':3}

T5_DECODING_STRAT = \
	{'top_p':0.85,
	 'min_length':10,
	 'max_length':30,
	 'repetition_penalty':4}

ALL_METRICS = ['BertSimilarity', 'EntitiesCount', 'GPT2Perplexity', 'KwCount', 'BleuScore', 'RougeScore']

SizeInfo = namedtuple('Size', 'inf_chars sup_chars mean_tokens token')
SMALL = SizeInfo(inf_chars=1, sup_chars=700, mean_tokens=100, token='[S]')
MEDIUM = SizeInfo(inf_chars=701, sup_chars=1200, mean_tokens=250, token='[M]')
LARGE = SizeInfo(inf_chars=1201, sup_chars=1700, mean_tokens=350, token='[L]')
SIZES = [SMALL, MEDIUM, LARGE]

def get_size_from_chars(length_in_chars):
	size = SMALL
	for s in SIZES[1:]:
		if length_in_chars >= s.inf_chars:
			size = s
	return size

def get_size_from_tokens(length_in_tokens):
	dists = [abs(s.mean_tokens - length_in_tokens) for s in SIZES]
	min_dist = min(dists)
	for i, s in enumerate(SIZES):
		if dists[i] == min_dist:
			return s

def text_batch_splitter(strings: List[str], max_length: int) -> Tuple[List[str], List[Tuple[int, int]]]:
	"""
	Takes a list of strings and split the ones of them that are longer than a specified length, growing the list
	where splitting is needed. Split information can then be used to merge the output information back after use.
	Cuts happen at whitespaces.
	:param strings: list of input strings of any length
	:param max_length: maximum length after which a string must be split into smaller ones
	:return new list of strings of size <= max_length, with long strings splitted in consecutive indices
	:return split information: indices of split strings, with number of consecutive elements involved (pass to merger later)
	"""
	new_strings = []
	split_information = []

	for full_str in strings:
		if len(full_str) <= max_length:
			new_strings.append(full_str)
		else:
			# Splitting the text in sequences the model can accept
			words = full_str.split()
			n_seqs = len(full_str) // max_length + 1
			seqs = []
			wi = 0
			for i in range(n_seqs):
				current = ""
				while wi < len(words) and len(current) + len(words[wi]) < len(full_str) / n_seqs:
					current += words[wi] + " "
					wi += 1
				seqs.append(current[:-1])

			split_information.append((len(new_strings), len(seqs)))
			new_strings += seqs

	return new_strings, split_information


def token_batch_splitter(inputs: List[str], max_length: int) -> Tuple[List[str], List[Tuple[int, int]]]:
	"""
	Takes a list of strings and split the ones of them that are longer than a specified length, growing the list
	where splitting is needed. Split information can then be used to merge the output information back after use.
	Cuts happen at whitespaces.
	:param inputs: list of input arrays of any length, each input containing tuples of (token, is valid position to cut)
	:param max_length: maximum length after which a string must be split into smaller ones
	:return new list of strings of size <= max_length, with long strings splitted in consecutive indices
	:return split information: indices of split strings, with number of consecutive elements involved (pass to merger later)
	"""
	new_inputs = []
	split_information = []

	for i, full_input in enumerate(inputs):
		if len(full_input) <= max_length:
			new_inputs.append(full_input)
		else:
			# Splitting the text in sequences the model can accept
			wi = 0
			current_seq = []
			last_word = 0
			split_seqs = []
			while wi < len(full_input):

				inp = full_input[wi]
				wi += 1

				current_seq.append(inp)

				if len(current_seq) > max_length:
					if inp[1] == 1:
						split_seqs.append(current_seq[:-1])
						current_seq = [inp]
					else:
						split_seqs.append(current_seq[:last_word])
						current_seq = current_seq[last_word:]
					last_word = 0
				elif inp[1] == 1:
					last_word = len(current_seq) - 1

			split_seqs.append(current_seq)
			split_information.append((len(new_inputs), len(split_seqs)))
			new_inputs += split_seqs

			assert sum(len(ss) for ss in split_seqs) == len(full_input)

			try:
				assert all(len(ss) <= max_length for ss in split_seqs)
			except AssertionError:
				for ss in split_seqs:
					if len(ss) > max_length:
						print(ss)

	return new_inputs, split_information


def batch_merger(outputs: List[Any], split_information: List[Tuple[int, int]], merge_function=None,
				 reduce_function=None, apply_on_single=False) -> List[Any]:
	"""
	Merges consecutive outputs related to previously split inputs. Taking a list of outputs, it builds a shorter
	list of outputs with some of them merged based on split information.
	:param outputs: list of outputs containing some outputs to merge
	:param split_information: indices of split outputs, with number of consecutive elements to merge
	:param merge_function: function to apply on every list of outputs to merge (must be None if using reduce_function)
	:param reduce_function: function to apply recursively on outputs to merge (must be None if using merge_function)
	:param apply_on_single: should the merge_function be applied even on non-split elements (must be False if using reduce_function)
	:return: a merged, eventually shorter, list of outputs
	"""
	assert merge_function is not None or reduce_function is not None
	assert merge_function is None or reduce_function is None
	assert reduce_function is None or not apply_on_single

	new_outputs = []
	split_indices = [x[0] for x in split_information]
	split_lengths = {i: s for (i, s) in split_information}

	i = 0
	while i < len(outputs):
		if i not in split_indices:
			out = outputs[i]
			if apply_on_single:
				out = merge_function([out])
			new_outputs.append(out)
			i += 1
		else:
			length = split_lengths[i]
			outs = outputs[i: i + length]

			if merge_function is None:
				new_outputs.append(functools.reduce(reduce_function, outs))
			else:
				new_outputs.append(merge_function(outs))

			i += length

	return new_outputs


def summary_selector(summary_models=None):
	"""
	Will create a function that take as input a dict of summaries :
	{'T5': [str] summary_generated_by_T5, ..., 'KW': [str] summary_generted_by_KW}
	and randomly return a summary that has been generated by one of the summary_model in summary_model

	if summary_models is none, will not use summaru
	:param summary_models: list of str(SummarizerModel)
	:return: function [dict] -> [str]
	"""
	if summary_models is None or len(summary_models) == 0 or \
		(len(summary_models) == 1 and summary_models[0] == ""):
		return lambda x: ""

	summary_model = random.choice(summary_models)
	return lambda summaries_dict: summaries_dict[summary_model]



def merge_summaries():
	"""
	Add summaries for each paragraph of every book.
	We merge the 4 distinct summaries obtained from the summarizers (T5, BART, PYSUM, KW) in our preproc data files
	These summaries are stored as json files (like preproc) in a different folder per summarizer
	The final format in preproc files is a {'summarizer_name':'summary',...}
	"""
	# Loop on all books. Look at original json file + corresponding files but with a summary (x4)
	files = os.listdir(PREPROC_PATH)
	for f in files:
		if PREPROC_SUFFIX in f:
			d_id = f[:-len(PREPROC_SUFFIX)]
			data = json.load(open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'r'))
			if os.path.exists(FOLDER_NAME_T5):
				data_t5 = json.load(open(FOLDER_NAME_T5 + PREFIX_T5 + d_id + PREPROC_SUFFIX, 'r'))
			if os.path.exists(FOLDER_NAME_BART):
				data_bart = json.load(open(FOLDER_NAME_BART + PREFIX_BART + d_id + PREPROC_SUFFIX, 'r'))
			if os.path.exists(FOLDER_NAME_PYSUM):
				data_pysum = json.load(open(FOLDER_NAME_PYSUM + PREFIX_PYSUM + d_id + PREPROC_SUFFIX, 'r'))
			if os.path.exists(FOLDER_NAME_BERTSUM):
				data_bertsum = json.load(open(FOLDER_NAME_BERTSUM + PREFIX_BERTSUM + d_id + PREPROC_SUFFIX, 'r'))
			if os.path.exists(FOLDER_NAME_KW):
				data_kw = json.load(open(FOLDER_NAME_KW + PREFIX_KW + d_id + PREPROC_SUFFIX, 'r'))

			# Add summary from each summariser to the original preproc json file
			for i in range(len(data['paragraphs'])):
				data['paragraphs'][i]['summaries'] = dict()
				if os.path.exists(FOLDER_NAME_T5):
					data['paragraphs'][i]['summaries'].update(data_t5['paragraphs'][i]['summaries'])
				if os.path.exists(FOLDER_NAME_BART):
					data['paragraphs'][i]['summaries'].update(data_bart['paragraphs'][i]['summaries'])
				if os.path.exists(FOLDER_NAME_PYSUM):
					data['paragraphs'][i]['summaries'].update(data_pysum['paragraphs'][i]['summaries'])
				if os.path.exists(FOLDER_NAME_BERTSUM):
					data['paragraphs'][i]['summaries'].update(data_bertsum['paragraphs'][i]['summaries'])
				if os.path.exists(FOLDER_NAME_KW):
					data['paragraphs'][i]['summaries'].update(data_kw['paragraphs'][i]['summaries'])

			# Save modifications to preproc json files
			json.dump(data, open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False,
					  indent=1)

def pad_left_side(sequences, padding_value):
	"""
	Modification of torch.nn.utils.rnn.pad_sequence so that we pad left side and not right side
	:param sequences : list of tensors
	:param padding_value : tokenizer.pad_token_id
	:return tensor of shape (len(sequences), max_length of sequence in sequences)
			the tensor are padded on the left side using pad_token_id from GPT2 tokenizer
	"""
	max_len = max([s.size(0) for s in sequences])
	out_dims = (len(sequences), max_len)
	out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
	for i, tensor in enumerate(sequences):
		length = tensor.size(0)
		out_tensor[i, max_len - length:] = tensor
	return out_tensor
