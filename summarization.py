"""
Script to apply summarization on the preproccess paragraph
"""

from src.flexible_models.flexible_sum import FlexibleSum, SummarizerModel
from src.utils import PREPROC_SUFFIX
from tqdm.notebook import tqdm
import os
import json
import argparse
import re

def apply_summarization(input_folder_path, output_folder_path, list_of_book_id, summarizer_model, batch_size=1):
	"""
	apply summarization on all json file of a given path
	:param input_folder_path: path to the folder containing the preprocess json file
	:param output_folder_path: path to the folder in which the new json will be dumped
	:param list_of_book_id: list[str] : list of book id to summaries
	:param summarizer_model: SummarizerModel value
	:param batch_size: batch size for T5 and BART
	"""
	summarizer = FlexibleSum(summarizer_model, batch_size)

	# Compute summary on each novel
	for book_id in tqdm(list_of_book_id):
		with open(input_folder_path+book_id+PREPROC_SUFFIX, 'r', encoding='utf-8') as f:
			data = json.load(f)
		print("Summarizing book : ", book_id)
		paragraphs = list(map(lambda x: x['text'], data['paragraphs']))
		summaries = summarizer(paragraphs)

		for i, summary in enumerate(summaries):
			if type(data['paragraphs'][i]['summaries']) == list:
				data['paragraphs'][i]['summaries'] = dict()
			data['paragraphs'][i]['summaries'][str(summarizer_model)] = summary

		json.dump(data, open(output_folder_path + str(summarizer_model) + '_' + book_id + PREPROC_SUFFIX,
		                     'w', encoding='utf-8'))


def retrieve_list_of_books_to_summarize(input_folder_path, output_folder_path, summarizer_model):
	"""
	Compare json files in input_folder and output_folder to spot files that has not been summarize yet by the
	summarizer 'summarizer_model'
	:param input_folder_path: path to the folder containing the preprocess json file
	:param output_folder_path: path to the folder in which the new json will be dumped
	:param summarizer_model: SummarizerModel value
	:return: list[str] list of book id not summarize yet
	"""
	# Create folder where data is stored if it does not exist
	if not os.path.exists(output_folder_path):
		os.mkdir(output_folder_path)

	# input_book_ids = set(re.search("(.*)" + PREPROC_SUFFIX, file).group(1) for file in os.listdir(input_folder_path))
	input_book_ids = set(re.search("(.*)" + PREPROC_SUFFIX, file).group(1)
	                     for file in os.listdir(input_folder_path)
	                     if re.search("(.*)" + PREPROC_SUFFIX, file) is not None)
	PREFIX_SUM = str(summarizer_model) + "_"
	summarized_book_ids = set(re.search(PREFIX_SUM+"(.*)"+PREPROC_SUFFIX, file).group(1)
	                          for file in os.listdir(output_folder_path)
	                          if re.search(PREFIX_SUM+"(.*)"+PREPROC_SUFFIX, file) is not None)

	print("Number of books already summarized with ", str(summarizer_model), " : ", len(summarized_book_ids))
	print("Number of books that still need to be summarized with ", str(summarizer_model), " : ",
	      len(input_book_ids - summarized_book_ids))

	return list(input_book_ids - summarized_book_ids)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_data_folder",
	                    type=str,
	                    required=True,
	                    help="The directory where the preprocess json files are stored")

	parser.add_argument("--output_data_folder",
	                    type=str,
	                    required=True,
	                    help="The directory where the preprocess json with summary will be stored")

	parser.add_argument("--T5", action="store_true", help="Use T5 summarizer")
	parser.add_argument("--BART", action="store_true", help="Use BART summarizer")
	parser.add_argument("--PYSUM", action="store_true", help="Use pysummarizer")
	parser.add_argument("--BERT_SUM", action="store_true", help="Use BERT summarizer")
	parser.add_argument("--KW", action="store_true", help='Use keywords summarization')
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size for T5 and BART")

	args = parser.parse_args()

	if args.T5:
		book_ids = retrieve_list_of_books_to_summarize(args.input_data_folder, args.output_data_folder, SummarizerModel.T5)
		apply_summarization(args.input_data_folder, args.output_data_folder, book_ids, SummarizerModel.T5, args.batch_size)

	if args.BART:
		book_ids = retrieve_list_of_books_to_summarize(args.input_data_folder, args.output_data_folder, SummarizerModel.BART)
		apply_summarization(args.input_data_folder, args.output_data_folder, book_ids, SummarizerModel.BART, args.batch_size)

	if args.BERT_SUM:
		book_ids = retrieve_list_of_books_to_summarize(args.input_data_folder, args.output_data_folder, SummarizerModel.BERT_SUM)
		apply_summarization(args.input_data_folder, args.output_data_folder, book_ids, SummarizerModel.BERT_SUM)

	if args.PYSUM:
		book_ids = retrieve_list_of_books_to_summarize(args.input_data_folder, args.output_data_folder, SummarizerModel.PYSUM)
		apply_summarization(args.input_data_folder, args.output_data_folder, book_ids, SummarizerModel.PYSUM)

	if args.KW:
		book_ids = retrieve_list_of_books_to_summarize(args.input_data_folder, args.output_data_folder, SummarizerModel.PYSUM)
		apply_summarization(args.input_data_folder, args.output_data_folder, book_ids, SummarizerModel.KW)
