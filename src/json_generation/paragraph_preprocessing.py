from src.utils import *
from src.flexible_models.paragraph_parser import ParagraphParser


def separate_paragraphs_all_files(overwrite: bool, files: List[str] = None, min_threshold: int = 20,
								  min_length: int = 600, max_length: int = 900, skip_begin=0.0, skip_end=0.0,
								  verbose: int = 1):
	"""
	Applies the paragraph separation process (next function) on all _novel.json files to produce _preproc.json files.
	:param overwrite: should already preprocessed files be re-preprocessed
	:param files: optional list of files to work on
	:param min_threshold: minimum length (in chars) a paragraph should be to be taken into account.
						  Lower than this threshold often means it's a title or a chapter separator.
	:param min_length: minimum length of final sub-paragraphs. It will not be strictly respected though.
	:param max_length: maximum length of final sub-paragraphs. Strictly respected.
	:param skip_begin: percent of the paragraphs to skip at the beginning of the books, to avoid non-relevant content
	:param skip_end: percent of the paragraphs to skip at the end of the books, to avoid non-relevant content
	:param verbose: 0 for silent execution, 1 for statistics and 2 for statistics and histogram of paragraphs sizes.
	"""
	parser = ParagraphParser(min_threshold, min_length, max_length)

	files = os.listdir(NOVEL_PATH) if files is None else [f + NOVEL_SUFFIX for f in files]
	treated_files = os.listdir(PREPROC_PATH)
	treated_ids = [f[:-len(PREPROC_SUFFIX)] for f in treated_files]

	# Loop over all stored texts, already converted json format but not fully pre-processed
	for f in files:
		d_id = f[:-len(NOVEL_SUFFIX)]
		if not os.path.exists(NOVEL_PATH + d_id + NOVEL_SUFFIX):
			continue
		if overwrite or d_id not in treated_ids:
			separate_in_paragraphs(parser, d_id, skip_begin, skip_end, verbose)


def separate_in_paragraphs(parser: ParagraphParser, d_id: str = None, skip_begin=0.0, skip_end=0.0, verbose: int = 2):
	"""
	Separates the text contained in a _novel.json file into sub-paragraphs of desired length in a _preproc.json file.
	It will try to preserve consistency by avoiding to merge different parts of the book and to cut sentences in the middle.
	:param parser: The paragraph parser instance to use.
	:param d_id: prefix to the _novel.json and _preproc.json files.
	:param skip_begin: percent of the paragraphs to skip at the beginning of the book, to avoid non-relevant content
	:param skip_end: percent of the paragraphs to skip at the end of the book, to avoid non-relevant content
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
	paragraphs = parser(full_text, novel_data['persons'], novel_data['organisations'], novel_data['locations'],
						novel_data['misc'], verbose)

	len_par = len(paragraphs)
	if skip_begin > 0:
		n_skip_begin = int(len_par * skip_begin)
		paragraphs = paragraphs[n_skip_begin:]
	if skip_end > 0:
		n_skip_end = int(len_par * skip_end)
		paragraphs = paragraphs[:-n_skip_end]

	if verbose >= 1:
		print("Skipped", len_par - len(paragraphs), "paragraphs.")

	# Saving JSON file
	novel_data.pop('text')
	novel_data.pop('persons')
	novel_data.pop('organisations')
	novel_data.pop('locations')
	novel_data.pop('misc')
	novel_data['paragraphs'] = paragraphs
	json.dump(data, open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
