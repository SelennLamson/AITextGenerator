#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:04:40 2020

@author: alexandreduval
"""

## SET UP 
# import nltk
# nltk.download('punkt') 


# Import libraries 
import json 
from tqdm import tqdm
from gutenberg.acquire import load_etext 
from gutenberg.cleanup import strip_headers 
from gutenberg.query import get_etexts, list_supported_metadatas, get_metadata
from sortedcontainers import SortedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import zlib # error message
from gutenberg._domain_model.exceptions import UnknownDownloadUriException # error message


# Define paths 
LOC = '../../data/metadata/'
NEW_LOC = '../../data/metadata/files/'
old_filename = 'clean_data.json'
new_filename = '1.json'

# Import json dataset 
try:
	data = json.load(open(LOC + old_filename, 'r'))
except UnicodeDecodeError:
	data = json.load(open(LOC + old_filename, 'r', encoding='utf-8'))


# Use Porter Stemmer
porter = PorterStemmer()
def stemSentence(sentence):
    token_words=word_tokenize(sentence) # tokenise sentence --> list of words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return "".join(stem_sentence)


# Define genres - particular format simply to facilitate pre processing with Stemmer
Genres = SortedDict({'adventure': 'adventure', 
            'biography': 'biography', 
            'children': 'children', 
            'detective': 'detective', 
            'english': 'english', 
            'fantasy': 'fantasy', 
            'fiction': 'fiction', 
            'history': 'history', 
            'horror': 'horror', 
            'mystery': 'mystery', 
            'romance': 'romance', 
            'satire': 'satire', 
            'science-fiction': 'science-fiction', 
            'short stories': 'short stories', 
            'teen': 'teen', 
            'thriller': 'thriller', 
            'western': 'western'})

# Stem Genres
for i in range(len(Genres)):
    Genres[list(Genres.keys())[i]] = stemSentence(list(Genres.values())[i])

# List of all genres appelation in Genre_bis  
Genres_l = [item for item in Genres.values()]


def FindRealTheme(l, Genres, Genres_l, new_el):
    """
    l is the list of keywords, gathered from gutenberg
    use it to retrieve genre/subject of the book
    """
    #  Process list of themes - flatten it to see all keywords
    flat_list = list(set([item.lower() for sublist in l for item in sublist.split()]))

    # If keywords in 'theme' correspond to a genre, add it
    genre = []
    for el in flat_list:
        el = stemSentence(el) # stemming of the genre 
        if el in Genres_l:# check if keyword correspond to a genre
            genre.append(Genres.keys()[Genres_l.index(el)]) # store true genre(s) (no stemming)
    new_el['theme'] = genre 
    return new_el
        

# Filter genres 
def FilterGenres(): 
    """
    simply to visualise
    """
    l = []
    for i in range(100):
        l.append(data[str(b_id)]['theme'])
    return l  
# FilterGenres()


def CreateJson(b_id, Genres, Genres_l): 
    # Set the id of the book you want to process 
    
    # Create new element that will be our new json file for the selected book - choose this way to avoid computationnally expensive storage
    # It will contain all information (text, id, author, title, genre) and strips useless info at beginning 
    new_el = data[str(b_id)]
    if 'en' in new_el['language']: # keep only english files
        new_el['id'] = str(b_id)
        new_el = FindRealTheme(new_el['theme'], Genres, Genres_l, new_el)
        try: 
            new_el['text'] = strip_headers(load_etext(b_id)).strip() 
        except (zlib.error, UnknownDownloadUriException):
            pass
        # Save it as a new json file 
        new_filename = '{}.json'.format(b_id)
        json.dump(new_el, open(NEW_LOC + new_filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


# FOR ALL DATAPOINTS 
# !!!!! Need to replace 'subject' by 'theme' in FindRealTheme method for the FINAL VERSION 
for i in tqdm(range(10500,10550)): 
    #print(i)
    b_id = i 
    CreateJson(b_id, Genres, Genres_l)

 



#################
# TRUE CODE
#################

"""
# Do it for all datapoints 
for i in range(1,59395): 
    b_id = i
    new_el = data[str(b_id)]
    if 'en' in new_el['language']: 
        new_el['id'] = str(b_id)
        new_el['text'] = strip_headers(load_etext(b_id)).strip() 
        new_filename = '{}.json'.format(i)
        json.dump(new_el, open(NEW_LOC + new_filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
"""

