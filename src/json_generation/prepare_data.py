"""
NER and split text into paragraphs
"""

from src.json_generation.ent_sum_preprocessing import perform_global_ner_on_all
from src.json_generation.paragraph_preprocessing import separate_paragraphs_all_files
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.utils import *


def prepare_data(files: List[str] = None, do_ner=True, do_split=True, verbose=0):
	"""
	Perform NER on the selected json files, stored in 'data/metadata/files/'. Save them into 'data/novel/'
	Split into paragraphs the full text of the selected files contained in 'data/novel/'
	and save them into 'data/preproc/'.
	:param files: list of the files ids (refer clean_data file) that will be preprocessed
	:param do_ner: do we perform NER or not
	:param do_split: do we split the book's full text into paragraphs or not
	:param verbose: print information
	"""
	if files is None:
		if do_ner:
			files = [f[:-len(METADATA_SUFFIX)] for f in os.listdir(METADATA_PATH)]
		else:
			files = [f[:-len(NOVEL_SUFFIX)] for f in os.listdir(NOVEL_PATH)]

	if do_ner:
		if verbose >= 1:
			print("Performing NER...")
		ner_model = FlexibleBERTNER(BERT_NER_LARGE, batch_size=256, max_length=128)
		perform_global_ner_on_all(ner_model, files, verbose=verbose)

	if do_split:
		if verbose >= 1:
			print("Splitting into paragraphs...")
		separate_paragraphs_all_files(True, files, min_length=1400, max_length=1700, skip_begin=0.03, verbose=verbose)

	if verbose >= 1:
		print("Preparation terminated.")
