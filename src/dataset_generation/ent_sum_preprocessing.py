#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 15 2020

@author: thomaslamson
"""

import json
import os
from src.utils import *


def prepare_json_templates(overwrite):
	files = os.listdir(PREPROC_PATH)
	treated_files = os.listdir(ENTSUM_PATH)
	treated_ids = [f[:-len(ENTSUM_SUFFIX)] for f in treated_files]

	for f in files:
		d_id = f[:-len(PREPROC_SUFFIX)]
		if not os.path.exists(PREPROC_PATH + d_id + PREPROC_SUFFIX):
			continue
		if overwrite or d_id not in treated_ids:

			# Reading JSON file
			data = json.load(open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'r'))
			novel_data = data['novel']
			paragraphs = novel_data['paragraphs']

			# Adding empty lists
			for p in paragraphs:
				p['persons'] = []
				p['locations'] = []
				p['organisations'] = []
				p['misc'] = []
				p['summaries'] = []

			# Saving JSON file
			json.dump(data, open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'w'))


def perform_ner_on_all(model, verbose=1):
	files = os.listdir(PREPROC_PATH)
	for f in files:
		d_id = f[:-len(PREPROC_SUFFIX)]
		if not os.path.exists(PREPROC_PATH + d_id + PREPROC_SUFFIX):
			continue
		print("Processing file:", f)
		perform_entity_recognition(model, d_id, verbose)


def perform_entity_recognition(model, d_id=None, verbose=1):
	# Input of file ID
	if d_id is None:
		while True:
			d_id = input("Select a novel id: ")
			if os.path.exists(ENTSUM_PATH + d_id + ENTSUM_SUFFIX):
				break
			print("ERROR - Id", d_id, "not found.")

	# Reading JSON file
	data = json.load(open(ENTSUM_PATH + d_id + ENTSUM_SUFFIX, 'r'))
	novel_data = data['novel']
	paragraphs = novel_data['paragraphs']

	total_p = len(paragraphs)
	current_percent = 0
	for pi, p in enumerate(paragraphs):
		if verbose >= 1:
			if int(pi / total_p * 100) > current_percent:
				current_percent = int(pi / total_p * 100)
				print("\rNER - {}%".format(current_percent), end="")

		text = p['text']

		# Splitting the text in sequences the model can accept
		words = text.split()
		n_seqs = len(text) // 2000 + 1
		seqs = []
		wi = 0
		for i in range(n_seqs):
			current = ""
			while len(current) < len(text) / n_seqs and wi < len(words):
				current += words[wi] + " "
				wi += 1
			seqs.append(current)

		# Performing inference on each sequence
		output = []
		for seq in seqs:
			output += model.predict(seq + ".")

		# Merging predictions together, using probability rule: p(A or B) = p(A) + p(B) - p(A)*p(B)
		entities = dict()
		current_entity = None
		current_confidence = 0
		current_tag = None
		for o in output:
			tag = o['tag'][2:]
			begin = o['tag'][0] == 'B'
			entity = o['word']
			confidence = o['confidence']

			# 1. If we encounter a new entity, but current one is not registered yet
			# OR
			# 2. We see no tag anymore but we had an entity in mind, so we register it
			if (tag != "" and begin and current_entity is not None) or \
			   (tag == "" and current_entity is not None):
				if current_entity in entities:
					# We already encountered this entity in a previous sequence
					prev_tag, prev_conf = entities[current_entity]
					if prev_tag == current_tag:
						# This is the same tag, we apply p(A or B) rule
						conf = prev_conf + current_confidence - prev_conf * current_confidence
						entities[current_entity] = (prev_tag, conf)
					elif prev_conf < current_confidence:
						# This is not the same tag as before, we just keep the best one
						entities[current_entity] = (current_tag, current_confidence)
				else:
					# This is the first time we encounter this entity
					entities[current_entity] = (current_tag, current_confidence)

				# After registering, we reset the entity to None
				current_entity = None

			# Now, we process the current tag
			if tag != "":	# We have a Named Entity
				if begin:	# It is a new one
					current_entity = entity
					current_confidence = confidence
					current_tag = tag
				elif current_entity is not None and not begin:	# It is continuing the current one
					current_entity += " " + entity
					current_confidence = current_confidence * 0.7 + confidence * 0.3  # Simple heuristic to merge confidences

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
