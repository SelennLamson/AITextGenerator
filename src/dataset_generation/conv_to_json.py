#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:00:21 2020

@author: alexandreduval
"""

# Constants
NOVEL_PATH = '../../data/novel/'
NOVEL_SUFFIX = '_novel.json'
RAW_PATH = '../../data/raw/'
RAW_SUFFIX = '.txt'

# Input of file ID
d_id = '#'
while True:
	d_id = input("Select a novel id: ")
	if os.path.exists(RAW_PATH + d_id + RAW_SUFFIX):
		break
	print("ERROR - Id", d_id, "not found.")

# Import text file
import json
filename = '3021.txt'
# filename = RAW_PATH + d_id + RAW_SUFFIX
dico = {} 
            
# Read text file
with open(filename, 'r') as f:
	reviews = f.read()
    
# Place text into a dico for json format 
dico["novel"] = {}
dico["novel"]["title"] = None 
dico["novel"]["author"] = None
dico["novel"]['theme'] = None 
dico["novel"]['text'] = reviews[:1000]

# Create a json file 
out_file = open("../../data/novel/3021_novel.json", "w") 
json.dump(dico, out_file, indent = 4, sort_keys = False) 
out_file.close() 



