#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

from typing import List, Any, Tuple
import functools
from math import ceil

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

DEFAULT_DECODING_STRATEGY = {
	'do_sample': True,
	'max_length': 50,
	'top_k': 50,
	'top_p': 0.95
}


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
