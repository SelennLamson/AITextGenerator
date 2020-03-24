#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

# Import libraries 
import json
import os
from src.utils import *
from src.flexible_models import *


def separate_paragraphs_all_files(overwrite:bool, min_threshold:int = 20, min_length:int = 600, max_length:int = 900, verbose:int = 1):
	"""
	Applies the paragraph separation process (next function) on all _novel.json files to produce _preproc.json files.
	:param overwrite: should already preprocessed files be re-preprocessed
	:param min_threshold: minimum length (in chars) a paragraph should be to be taken into account.
						  Lower than this threshold often means it's a title or a chapter separator.
	:param min_length: minimum length of final sub-paragraphs. It will not be strictly respected though.
	:param max_length: maximum length of final sub-paragraphs. Strictly respected.
	:param verbose: 0 for silent execution, 1 for statistics and 2 for statistics and histogram of paragraphs sizes.
	"""
	parser = ParagraphParser(min_threshold, min_length, max_length)

	files = os.listdir(NOVEL_PATH)
	treated_files = os.listdir(PREPROC_PATH)
	treated_ids = [f[:-len(PREPROC_SUFFIX)] for f in treated_files]

	# Loop over all stored texts, already converted json format but not fully pre-processed
	for f in files:
		d_id = f[:-len(NOVEL_SUFFIX)]
		if not os.path.exists(NOVEL_PATH + d_id + NOVEL_SUFFIX):
			continue
		if overwrite or d_id not in treated_ids:
			separate_in_paragraphs(parser, d_id, verbose)


def separate_in_paragraphs(parser: ParagraphParser, d_id:str = None, verbose:int = 2):
	"""
	Separates the text contained in a _novel.json file into sub-paragraphs of desired length in a _preproc.json file.
	It will try to preserve consistency by avoiding to merge different parts of the book and to cut sentences in the middle.
	:param parser: The paragraph parser instance to use.
	:param d_id: prefix to the _novel.json and _preproc.json files.
	:param verbose: 0 for silent execution, 1 for statistics and 2 for statistics and histogram of paragraphs sizes.
	"""
	# Input of file ID
	if d_id is None:
		while True:
			d_id = input("Select a novel id: ")
			if os.path.exists(NOVEL_PATH + d_id + NOVEL_SUFFIX):
				break
			print("ERROR - Id", d_id, "not found.")

	# Reading JSON file
	try:
		data = json.load(open(NOVEL_PATH + d_id + NOVEL_SUFFIX, 'r'))
	except UnicodeDecodeError:
		data = json.load(open(NOVEL_PATH + d_id + NOVEL_SUFFIX, 'r', encoding='utf-8'))
	# novel_data = data['novel']
	novel_data = data
	full_text = novel_data['text']

	if verbose >= 1:
		print("\n--- NOVEL DATA ---")
		print("Title:\t", novel_data['title'])
		print("Author:\t", novel_data['author'])
		print("Theme:\t", novel_data['theme'])
	paragraphs = parser(full_text, verbose)

	# Saving JSON file
	novel_data.pop('text')
	novel_data['paragraphs'] = paragraphs
	json.dump(data, open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
