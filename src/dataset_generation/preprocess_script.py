import json
import os
import re

# Constants
NOVEL_PATH = '../../data/novel/'
NOVEL_SUFFIX = '_novel.json'
PREPROC_PATH = '../../data/preproc/'
PREPROC_SUFFIX = '_preproc.json'
MIN_LENGTH = 200
MAX_LENGTH = 500

# Input of file ID
d_id = '#'
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
novel_data = data['novel']
full_text = novel_data['text']
paragraphs = []

print("\n--- NOVEL DATA ---")
print("Title:\t", novel_data['title'])
print("Author:\t", novel_data['author'])
print("Theme:\t", novel_data['theme'])
print("Text:\t", full_text[:100].replace('\n', ' ') + "...")

# Parsing paragraphs
def add_paragraph(content):
	paragraphs.append({
		'size': len(content),
		'text': content
	})

# Removing any isolated line-breaks, any multiple whitespaces and separating in real paragraphs
striped_of_linebreaks = ''.join('\n' if elt == '' else elt for elt in full_text.split('\n'))
striped_of_multispaces = re.sub(r'[ ]+', ' ', striped_of_linebreaks)
real_paragraphs = [elt for elt in striped_of_multispaces.split('\n') if elt != '']

for rp in real_paragraphs:
	# Splitting into groups of sentences above MIN_LENGTH
	sentences = [s.strip() for s in rp.split('.')]
	current_paragraphs = []
	current = ''
	for i, s in enumerate(sentences):
		current += s + '. '
		if len(current) > MIN_LENGTH or i == len(sentences) - 1:
			current = current[:-1]
			current_paragraphs.append(current)
			current = ''

	# Cutting group of sentences in the middle if bigger than MAX_LENGTH, propagating words to next group
	for i, p in enumerate(current_paragraphs):
		if i == len(current_paragraphs) - 1:
			break
		if len(p) >= MAX_LENGTH:
			words = p.split(' ')
			new_p = ''
			wi = 0
			while len(new_p) + len(words[wi]) <= MAX_LENGTH:
				new_p += words[wi] + ' '
				wi += 1
			current_paragraphs[i] = new_p
			current_paragraphs[i+1] = ' '.join(words[wi:]) + ' ' + current_paragraphs[i+1]

	for p in current_paragraphs:
		if 0.5 * MIN_LENGTH <= len(p) <= MAX_LENGTH:
			add_paragraph(p)

# Saving JSON file
novel_data.pop('text')
novel_data['paragraphs'] = paragraphs
json.dump(data, open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'w'))

# Printing results
sizes = []
for p in paragraphs:
	sizes.append(p['size'])

print("\n--- EXTRACTED DATA ---")
print('Paragraphs:\t\t\t', len(sizes))
print('Average length:\t\t', int(sum(sizes) / len(sizes)))
print('Max length:\t\t\t', max(sizes))
print('Min length:\t\t\t', min(sizes))
print('% of raw text:\t\t {:.2f}'.format(100 * sum(sizes) / len(full_text)))
print('% of stripped text:\t {:.2f}'.format(100 * sum(sizes) / len(striped_of_multispaces)))
print('\n-- First paragraph:\n"' + paragraphs[0]['text'][:100] + '..."')
