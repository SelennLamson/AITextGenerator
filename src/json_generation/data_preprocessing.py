#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:04:40 2020

@author: alexandreduval
"""

# Import libraries 
from src.utils import *
import json 
from tqdm import tqdm
from gutenberg.acquire import load_etext 
from gutenberg.cleanup import strip_headers 
from gutenberg.query import get_etexts, list_supported_metadatas, get_metadata
from sortedcontainers import SortedDict
import zlib  # error message
from gutenberg._domain_model.exceptions import UnknownDownloadUriException  # error message


class DataPrepro:
	"""
	Creates a json file for each book contains the text of book as well as its related metadata
	"""

	def __init__(self):
		self.old_filename = 'clean_data.json'
		# Define literature genres - assign them keywords to be able to spot the genre from the theme of the book
		self.genres = SortedDict({
				'adventure':['action', 'adventure', 'adventures'],
				'biography': ['biography', 'autobiography'],
				'children': ['child', 'children', 'baby', 'fairy'],
				'detective': ['detective', 'crime', 'crimes', 'detectives', 'murder', 'police'],
				'english':['literature', 'literary'],
				'fantasy':['fantasy'],
				'fiction':['fiction', 'fictions'],
				'history': ['history', 'historical', 'historical-fiction'],
				'horror':['horror', 'paranormal', 'ghost'],
				'mystery':['mystery', 'mysteries'],
				'romance':['romance', 'love'],
				'satire': ['satirical', 'satire', 'criticism', 'satires'],
				'short stories': ['short-stories', 'tale', 'tales', 'short'],
				'science-fiction': ['sci-fy', 'science-fiction'],
				'teen':['teenager', 'teen', 'teenage', 'young', 'juvenile'],
				'thriller':['suspense', 'thriller', 'thrillers'],
				'western': ['western']})
		# Import json dataset 
		try:
			self.data = json.load(open(METADATA_ROOT + self.old_filename, 'r'))
		except UnicodeDecodeError:
			self.data = json.load(open(METADATA_ROOT + self.old_filename, 'r', encoding='utf-8'))

	def find_real_genre(self, l, genres, new_el):
		"""
		:param l: list of sentences indicating the theme of the book, gathered form gutenberg
		:param genres: dictionnary with keys: genres, keys: keywords to detect genre
		:param new_el: book considered (dictionnary)
		Find the genre(s) of a book from gutenberg theme information
		"""
		#  Process list of themes - flatten it to see all keywords
		flat_list = list(set([item.lower() for sublist in l for item in sublist.split()]))

		# If keywords in 'theme' correspond to a genre, store it
		genre = []
		for el in flat_list:
			for i,sublist in enumerate(list(genres.values())):
				if el in sublist:
					genre.append(genres.keys()[i])
					genre = list(set(genre))
		# Create a new key with the genre(s) of the book
		new_el['genre'] = genre
		return new_el

	def create_json(self, b_id):
		"""
		:param b_id: id of the book
		Create json file for each book (b_id)
		Don't for books where metadata is not accessible, cannot strip headers, is not in english, does not have a genre
		"""
		genres = self.genres

		# Create new element that will be our new json file for the selected book -
		# choose this way to avoid computationally expensive storage
		# It will contain all information (text, id, author, title, genre, theme) and strips useless info at beginning
		new_el = self.data[str(b_id)]
		if 'en' in new_el['language']: # keep only english files
			new_el['id'] = str(b_id) # keep its id
			new_el = self.find_real_genre(new_el['theme'], genres, new_el) # find genre
			self.data[str(b_id)]['genre'] = new_el['genre'] # add genre to original document
			try:
				new_el['text'] = strip_headers(load_etext(b_id)).strip()
				if len(new_el['genre']) != 0: # do not include these files (not relevant for our study)
					# Save it as a new json file if it is a novel
					new_filename = str(b_id)
					json.dump(new_el, open(METADATA_PATH + new_filename + METADATA_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
			except (zlib.error, UnknownDownloadUriException): # deal with errors when importing text or removing headers
				# print('exception')
				pass


