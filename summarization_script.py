from src.flexible_models.flexible_sum import FlexibleSum, SummarizerModel
from tqdm.notebook import tqdm
import os
import json
import argparse

"""
Script to apply summarization on the preproccess paragraph 
"""

def apply_summarization(input_folder_path, output_folder_path, summarizer_model, batch_size=1):
    """
    apply summarization on all json file of a given path
    :param input_folder_path: path to the folder containing the preprocess json file
    :param output_folder_path: path to the folder in which the new json will be dumped
    :param summarizer_model: SummarizerModel value
    :param batch_size: batch size for T5 and BART
    """
    if False:
        summarizer = FlexibleSum(summarizer_model, batch_size)
    json_files = [json_file for json_file in os.listdir(input_folder_path) if json_file[-4:] == "json"]

    # Compute summary on each novel
    for json_file_name in tqdm(json_files):
        with open(input_folder_path+json_file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("Summarizing book : ", json_file_name)
        paragraphs = list(map(lambda x: x['text'], data['paragraphs']))
        if False:
            summaries = summarizer(paragraphs)
        else:
            summaries = paragraphs
        for i, summary in enumerate(summaries):
            if type(data['paragraphs'][i]['summaries']) == list:
                data['paragraphs'][i]['summaries'] = dict()
            data['paragraphs'][i]['summaries'][str(summarizer_model)] = summary

        json.dump(data, open(output_folder_path+str(summarizer_model)+'_'+json_file_name, 'w', encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder",
                        type=str,
                        required=True,
                        help="The directory where the preprocess json files are stored")

    parser.add_argument("--output_data_forder",
                        type=str,
                        required=True,
                        help="The directory where the preprocess json with summary will be stored")

    parser.add_argument("--T5", action="store_true", help="Use T5 summarizer")
    parser.add_argument("--BART", action="store_true", help="Use BART summarizer")
    parser.add_argument("--PYSUM", action="store_true", help="Use pysummarizer")
    parser.add_argument("--BERT_SUM", action="store_true", help="Use BERT summarizer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for T5 and BART")

    args = parser.parse_args()

    if args.T5:
        apply_summarization(args.input_data_folder, args.output_data_folder, SummarizerModel.T5, args.batch_size)

    if args.BART:
        apply_summarization(args.input_data_folder, args.output_data_folder, SummarizerModel.BART, args.batch_size)

    if args.BERT_SUM:
        apply_summarization(args.input_data_folder, args.output_data_folder, SummarizerModel.BERT_SUM)

    if args.PYSUM:
        apply_summarization(args.input_data_folder, args.output_data_folder, SummarizerModel.PYSUM)
