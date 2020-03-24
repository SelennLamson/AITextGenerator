#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

from typing import List, Any, Tuple
import functools

# Constants
NOVEL_PATH = 'data/metadata/files/'
NOVEL_SUFFIX = '.json'
PREPROC_PATH = 'data/preproc/'
PREPROC_SUFFIX = '_preproc.json'
ENTSUM_PATH = 'data/ent_sum/'
ENTSUM_SUFFIX = '_entsum.json'

BERT_NER_LARGE = 'models/entity_recognition/BERT_NER_Large/'

def text_batch_splitter(strings:List[str], max_length:int) -> Tuple[List[str], List[Tuple[int, int]]]:
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
				while wi < len(words) and len(current) + len(words[wi]) < len(full_str) / n_seqs :
					current += words[wi] + " "
					wi += 1
				seqs.append(current[:-1])

			split_information.append((len(new_strings), len(seqs)))
			new_strings += seqs

	return new_strings, split_information


def batch_merger(outputs: List[Any], split_information:List[Tuple[int, int]], merge_function=None, reduce_function=None, apply_on_single=False) -> List[Any]:
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
			outs = outputs[i : i + length]

			if merge_function is None:
				new_outputs.append(functools.reduce(reduce_function, outs))
			else:
				new_outputs.append(merge_function(outs))

			i += length

	return new_outputs
