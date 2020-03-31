#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

import json
import os
import time
from typing import List, Dict, Tuple
from src.utils import *
from src.flexible_models import *


def perform_summarization_on_all(models: List[FlexibleSummarizer], files: List[str] = None, replace=False, verbose=1):
	"""
	Applies summarization models to all _ent_sum.json file and their paragraphs.
	:param models: a list of models to apply, all of them having a predict(text) method.
	:param files: optional list of files to work on
	:param replace: True to erase existing summaries and replace them, False (default) to add to existing ones.
	:param max_length: the maximum text length the model can accept. Paragraphs above that will be split and results will be merged.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""
	files = os.listdir(PREPROC_PATH) if files is None else [f + PREPROC_SUFFIX for f in files]
	for f in files:
		d_id = f[:-len(PREPROC_SUFFIX)]
		if not os.path.exists(PREPROC_PATH + d_id + PREPROC_SUFFIX):
			continue
		if verbose >= 1:
			print("Processing file:", f)
		add_summaries(models, replace, d_id, verbose)


def add_summaries(models: List[FlexibleSummarizer], replace=False, d_id=None, verbose=1):
	"""
	Applies summarization model to an _ent_sum.json file for each of its paragraphs.
	:param models: a list of models to apply, all of them having a predict(text) method.
	:param replace: True to erase existing summaries and replace them, False (default) to add to existing ones.
	:param d_id: prefix to the _entsum.json file.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""

	# Input of file ID
	if d_id is None:
		while True:
			d_id = input("Select a novel id: ")
			if os.path.exists(PREPROC_PATH + d_id + PREPROC_SUFFIX):
				break
			print("ERROR - Id", d_id, "not found.")

	# Reading JSON file
	data = json.load(open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'r'))
	# novel_data = data['novel']
	novel_data = data
	paragraphs = novel_data['paragraphs']

	total_p = len(paragraphs)
	current_percent = 0
	for pi, p in enumerate(paragraphs):
		if verbose >= 1:
			if int(pi / total_p * 100) > current_percent:
				current_percent = int(pi / total_p * 100)
				print("\rSUMMARIZATION - {}%".format(current_percent), end="")

		text = p['text']
		if replace:
			p['summaries'] = []

		# Applying the different summarization models to the paragraph
		for model in models:
			summary = model(text)

			summary = summary.replace('\n', ' ').strip()
			if summary not in p['summaries'] and summary != '':
				p['summaries'].append(summary)

	# Saving JSON file
	json.dump(data, open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
	if verbose >= 1:
		print("\rSUMMARIZATION - 100%")


def perform_global_ner_on_all(model: FlexibleBERTNER, files: List[str] = None, verbose:int = 1):
	"""
	Applies NER model on all metadata/file/id.json files, replacing already existing entities in the way.
	:param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
	:param files: optional list of files to work on
	:param verbose: 0 for silent execution, 1 to display progress.
	"""
	files = os.listdir(METADATA_PATH) if files is None else [f + METADATA_SUFFIX for f in files]
	for f in files:
		d_id = f[:-len(METADATA_SUFFIX)]
		if not os.path.exists(METADATA_PATH + d_id + METADATA_SUFFIX):
			continue
		if verbose >= 1:
			print("Processing file:", f)

		now = time.time()
		perform_global_ner_on_file(model, d_id, verbose)
		print("Time elapsed: {}s".format(int(time.time() - now)))


def perform_global_ner_on_file(model: FlexibleBERTNER, d_id:str = None, verbose:int = 1):
	"""
	Applies NER model on all a metadata/file/id.json file, replacing already existing entities in the way.
	:param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
	:param d_id: file id.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""

	# Input of file ID
	if d_id is None:
		while True:
			d_id = input("Select a novel id: ")
			if os.path.exists(METADATA_PATH + d_id + METADATA_SUFFIX):
				break
			print("ERROR - Id", d_id, "not found.")

	# Reading JSON file
	novel_data = json.load(open(METADATA_PATH + d_id + METADATA_SUFFIX, 'r', encoding='utf-8'))
	text = novel_data['text'].replace('\n', ' ')

	output = model.predict_with_index(text, verbose)

	persons = dict()
	locations = dict()
	organisations = dict()
	misc = dict()
	for pi, (index, entity, tag) in enumerate(output):
		if verbose >= 1:
			print("\rNER outputs - {:.2f}%".format(pi / len(output) * 100), end="")

		if tag == "PER":
			persons[index] = entity
		elif tag == "LOC":
			locations[index] = entity
		elif tag == "ORG":
			organisations[index] = entity
		elif tag == "MISC":
			misc[index] = entity

	novel_data['persons'] = persons
	novel_data['locations'] = locations
	novel_data['organisations'] = organisations
	novel_data['misc'] = misc

	# Saving JSON file
	json.dump(novel_data, open(NOVEL_PATH + d_id + NOVEL_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
	if verbose >= 1:
		print("\rNER outputs - 100%")

