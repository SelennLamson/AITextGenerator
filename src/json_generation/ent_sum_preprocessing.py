"""
Script to apply NER and summarization on the pre-proccessed paragraph
"""

import time
from tqdm import tqdm
import os
import json
import re
from typing import List, Dict, Tuple

from src.utils import *
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.flexible_models.flexible_sum import FlexibleSum, SummarizerModel


def perform_global_ner_on_all(model: FlexibleBERTNER, files: List[str] = None, verbose: int = 1):
    """
    Applies NER model on all metadata/file/id.json files, replacing already existing entities in the way.
    :param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
    :param files: optional list of files to work on
    :param verbose: 0 for silent execution, 1 to display progress.
    """
    files = os.listdir(METADATA_PATH) if files is None else [f + METADATA_SUFFIX for f in files]
    for f in files:
        d_id = f[:-len(METADATA_SUFFIX)]
        if not os.path.exists(METADATA_PATH + d_id + METADATA_SUFFIX):
            continue
        if verbose >= 1:
            print("Processing file:", f)

        now = time.time()
        perform_global_ner_on_file(model, d_id, verbose)
        print("Time elapsed: {}s".format(int(time.time() - now)))


def perform_global_ner_on_file(model: FlexibleBERTNER, d_id: str = None, verbose: int = 1):
    """
    Applies NER model on all a metadata/file/id.json file, replacing already existing entities in the way.
    :param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
    :param d_id: file id.
    :param verbose: 0 for silent execution, 1 to display progress.
    """

    # Input of file ID
    if d_id is None:
        while True:
            d_id = input("Select a novel id: ")
            if os.path.exists(METADATA_PATH + d_id + METADATA_SUFFIX):
                break
            print("ERROR - Id", d_id, "not found.")

    # Reading JSON file
    novel_data = json.load(open(METADATA_PATH + d_id + METADATA_SUFFIX, 'r', encoding='utf-8'))
    text = novel_data['text'].replace('\n', ' ')

    output = model.predict_with_index(text, verbose)

    persons = dict()
    locations = dict()
    organisations = dict()
    misc = dict()
    for pi, (index, entity, tag) in enumerate(output):
        if verbose >= 1:
            print("\rNER outputs - {:.2f}%".format(pi / len(output) * 100), end="")

		e = entity.strip()
		if len(e) > 2 and e not in ['The', 'the']:
			if tag == "PER":
				persons[index] = e
			elif tag == "LOC":
				locations[index] = e
			elif tag == "ORG":
				organisations[index] = e
			elif tag == "MISC":
				misc[index] = e

    novel_data['persons'] = persons
    novel_data['locations'] = locations
    novel_data['organisations'] = organisations
    novel_data['misc'] = misc

    # Saving JSON file
    json.dump(novel_data, open(NOVEL_PATH + d_id + NOVEL_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if verbose >= 1:
        print("\rNER outputs - 100%")


def apply_summarization(input_folder_path, output_folder_path, list_of_book_id, summarizer_model, batch_size=1):
    """
    apply summarization on all json file of a given path
    :param input_folder_path: path to the folder containing the preprocess json file
    :param output_folder_path: path to the folder in which the new json will be dumped
    :param list_of_book_id: list[str] : list of book id to summarize
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
