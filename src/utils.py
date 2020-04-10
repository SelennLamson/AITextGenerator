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
FOLDER_NAME_PYSUM = 'data/Preproc_PYSUM/'
PREFIX_PYSUM = 'PYSUM_'

WEBSERVICE_FEEDBACK = 'data/webservice_feedback/'

DEFAULT_DECODING_STRATEGY = {
    'do_sample': True,
    'min_length': 0,
    'max_length': 50,
    'top_k': 50,
    'top_p': 0.95
}


SizeInfo = namedtuple('Size', 'inf_chars sup_chars mean_tokens token')
SMALL = SizeInfo(inf_chars=1, sup_chars=700, mean_tokens=100, token='[S]')
MEDIUM = SizeInfo(inf_chars=701, sup_chars=1200, mean_tokens=250, token='[M]')
LARGE = SizeInfo(inf_chars=1201, sup_chars=1700, mean_tokens=350, token='[L]')
SIZES = [SMALL, MEDIUM, LARGE]

GPT2_BLOCK_SIZE = 1020

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
    if summary_models is None or len(summary_models) == 0:
        return lambda x: ""

    summary_model = random.choice(summary_models)
    return lambda summaries_dict: summaries_dict[summary_model]
