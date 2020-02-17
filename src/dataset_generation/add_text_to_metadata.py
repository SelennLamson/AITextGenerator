#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:04:40 2020

@author: alexandreduval
"""

# Import libraries 
import json 
from gutenberg.acquire import load_etext 
from gutenberg.cleanup import strip_headers 

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


###############
# EXAMPLE 
###############

# Set the id of the book you want to process 
b_id = 1

# Create new element that will be our new json file for the selected book - choose this way to avoid computationnally expensive storage
# It will contain all information (text, id, author, title, genre) and strips useless info at beginning 
new_el = data[str(b_id)]
new_el['id'] = str(b_id)
new_el['text'] = strip_headers(load_etext(b_id)).strip() 

# Save it as a new json file 
json.dump(new_el, open(NEW_LOC + new_filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


#################
# TRUE CODE
#################

"""
# Do it for all datapoints 
for i in range(1,59395): 
    b_id = i
    new_el = data[str(b_id)]
    new_el['id'] = str(b_id)
    new_el['text'] = strip_headers(load_etext(b_id)).strip() 
    new_filename = '{}.json'.format(i)
    json.dump(new_el, open(NEW_LOC + new_filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
"""