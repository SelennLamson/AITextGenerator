#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:04:35 2020

@author: alexandreduval
"""

# Set up 
#pip install gutenberg
#git clone https://github.com/c-w/Gutenberg.github
#cd Gutenberg
#pip install r-requirements-dev.pip
#brew install berkeley-db4
#pip install .


# Import libraries 
from gutenberg.acquire import load_etext, get_metadata_cache
from gutenberg.cleanup import strip_headers 
from gutenberg.query import get_etexts, list_supported_metadatas, get_metadata


# Download a text 
text = strip_headers(load_etext(2701)).strip()
print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'

# Populate the metadata cache - long operation 
cache = get_metadata_cache()
cache.populate()

# Check what kind of metadata we can get
print(list_supported_metadatas()) # prints (u'author', u'formaturi', u'language', ...)

# Extract metadata from that file
book_id = 1000 
print(get_metadata('title', book_id))  
print(get_metadata('author', book_id)) 
print(get_metadata('subject', book_id)) 
print(get_metadata('language', book_id))

# Find book id wrt to some filters
print(get_etexts('title', 'Moby Dick; Or, The Whale'))  # prints frozenset([2701, ...])
print(get_etexts('subject', 'Fiction'))        # prints frozenset([2701, ...])
print(get_etexts('author', 'Melville, Herman'))

"""
# Cache with more control 
from gutenberg.acquire.metadata import SqliteMetadataCache
cache = SqliteMetadataCache('/my/custom/location/cache.sqlite')
cache.populate()
set_metadata_cache(cache)
"""