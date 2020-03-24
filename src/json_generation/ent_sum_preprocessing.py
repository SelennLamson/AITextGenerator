#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

import json
import os
from typing import List, Dict, Tuple
from src.utils import *
from src.flexible_models import *


def prepare_json_templates(overwrite):
	"""
	Prepares _entsum.json files from _preproc.json files. It simply adds empty lists to each paragraph: persons, locations, organisations, misc and summaries.
	:param overwrite: Should already existing files be overwritten?
	"""
	files = os.listdir(PREPROC_PATH)
	treated_files = os.listdir(ENTSUM_PATH)
	treated_ids = [f[:-len(ENTSUM_SUFFIX)] for f in treated_files]

	for f in files:
		d_id = f[:-len(PREPROC_SUFFIX)]
		if not os.path.exists(PREPROC_PATH + d_id + PREPROC_SUFFIX):
			continue
		if overwrite or d_id not in treated_ids:

			# Reading JSON file
			print(d_id)
			data = json.load(open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'r'))
			# novel_data = data['novel']
			novel_data = data
			paragraphs = novel_data['paragraphs']

			# Adding empty lists
			for p in paragraphs:
				p['persons'] = []
				p['locations'] = []
				p['organisations'] = []
				p['misc'] = []
				p['summaries'] = []

			# Saving JSON file
			json.dump(data, open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'w', encoding = 'utf-8'), ensure_ascii = False, indent = 1)


def perform_summarization_on_all(models: List[FlexibleSummarizer], replace=False, verbose=1):
	"""
	Applies summarization models to all _ent_sum.json file and their paragraphs.
	:param models: a list of models to apply, all of them having a predict(text) method.
	:param replace: True to erase existing summaries and replace them, False (default) to add to existing ones.
	:param max_length: the maximum text length the model can accept. Paragraphs above that will be split and results will be merged.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""
	files = os.listdir(ENTSUM_PATH)
	for f in files:
		d_id = f[:-len(ENTSUM_SUFFIX)]
		if not os.path.exists(ENTSUM_PATH + d_id + ENTSUM_SUFFIX):
			continue
		if verbose >= 1:
			print("Processing file:", f)
		add_summaries(models, replace, d_id, max_length, verbose)


def add_summaries(models: List[FlexibleSummarizer], replace=False, d_id=None, max_length=2000, verbose=1):
	"""
	Applies summarization model to an _ent_sum.json file for each of its paragraphs.
	:param models: a list of models to apply, all of them having a predict(text) method.
	:param replace: True to erase existing summaries and replace them, False (default) to add to existing ones.
	:param d_id: prefix to the _entsum.json file.
	:param max_length: the maximum text length the models can accept. Paragraphs above that will be split and results will be merged.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""

	# Input of file ID
	if d_id is None:
		while True:
			d_id = input("Select a novel id: ")
			if os.path.exists(ENTSUM_PATH + d_id + ENTSUM_SUFFIX):
				break
			print("ERROR - Id", d_id, "not found.")

	# Reading JSON file
	data = json.load(open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'r'))
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
	json.dump(data, open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'w'))
	if verbose >= 1:
		print("\rSUMMARIZATION - 100%")


def perform_ner_on_all(model: FlexibleBERTNER, verbose:int = 1):
	"""
	Applies NER model to all _entsum.json and their paragraphs, replacing already existing entities in the way.
	:param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""
	files = os.listdir(ENTSUM_PATH)
	for f in files:
		d_id = f[:-len(ENTSUM_SUFFIX)]
		if not os.path.exists(ENTSUM_PATH + d_id + ENTSUM_SUFFIX):
			continue
		if verbose >= 1:
			print("Processing file:", f)
		perform_ner_on_file(model, d_id, verbose)


def perform_ner_on_file(model: FlexibleBERTNER, d_id:str = None, verbose:int = 1):
	"""
	Applies NER model to a _entsum.json file for each paragraph, replacing already existing entities in the way.
	:param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
	:param d_id: prefix to the _entsum.json file.
	:param verbose: 0 for silent execution, 1 to display progress.
	"""

	# Input of file ID
	if d_id is None:
		while True:
			d_id = input("Select a novel id: ")
			if os.path.exists(ENTSUM_PATH + d_id + ENTSUM_SUFFIX):
				break
			print("ERROR - Id", d_id, "not found.")

	# Reading JSON file
	data = json.load(open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'r'))
	# novel_data = data['novel']
	novel_data = data
	paragraphs = novel_data['paragraphs']

	total_p = len(paragraphs)
	current_percent = 0
	for pi, p in enumerate(paragraphs):
		if verbose >= 1:
			if int(pi / total_p * 100) > current_percent:
				current_percent = int(pi / total_p * 100)
				print("\rNER - {}%".format(current_percent), end="")

		print('Before NER')
		# Performing NER
		entities = model([p['text']])[0]

		# Registering inferred data to JSON file
		persons = []
		locations = []
		organisations = []
		misc = []
		for ent, (tag, confidence) in entities.items():
			if tag == "PER":
				persons.append(ent)
			elif tag == "LOC":
				locations.append(ent)
			elif tag == "ORG":
				organisations.append(ent)
			elif tag == "MISC":
				misc.append(ent)
		p['persons'] = persons
		p['locations'] = locations
		p['organisations'] = organisations
		p['misc'] = misc

	# Saving JSON file
	json.dump(data, open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'w'))
	if verbose >= 1:
		print("\rNER - 100%")

