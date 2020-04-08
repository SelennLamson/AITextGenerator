from src.flexible_models.flexible_sum import FlexibleSum, SummarizerModel
from tqdm import tqdm
import os
import json
import argparse

"""
Script to apply summarization on the preproccess paragraph 
"""

def apply_summarization(folder_path, summarizer_model, batch_size=1):
    """
    apply summarization on all json file of a given path
    :param folder_path: path to the model containing the preprocess json file
    :param summarizer_model: SummarizerModel value
    :param batch_size: batch size for T5 and BART
    """
    summarizer = FlexibleSum(summarizer_model, batch_size)
    json_files = [json_file for json_file in os.listdir(folder_path) if json_file[-4:] == "json"]

    # Compute summary on each novel
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        paragraphs = list(map(lambda x: x['text'], data['paragraphs']))

        summaries = summarizer(paragraphs)
        for i, summary in enumerate(summaries):
            data['paragraphs'][i]['summaries'][str(summarizer_model)] = summary

        json.dump(data, open(json_file, 'w', encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder",
                        type=str,
                        required=True,
                        help="The directory where the preprocess json files are stored")

    parser.add_argument("--T5", action="store_true", help="Use T5 summarizer")
    parser.add_argument("--BART", action="store_true", help="Use BART summarizer")
    parser.add_argument("--PYSUM", action="store_true", help="Use pysummarizer")
    parser.add_argument("--BERT_SUM", action="store_true", help="Use BERT summarizer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for T5 and BART")

    args = parser.parse_args()

    if args.T5:
        apply_summarization(args.data_folder, SummarizerModel.T5, args.batch_size)

    if args.BART:
        apply_summarization(args.data_folder, SummarizerModel.BART, args.batch_size)

    if args.BERT_SUM:
        apply_summarization(args.data_folder, SummarizerModel.BERT_SUM)

    if args.PYSUM:
        apply_summarization(args.data_folder, SummarizerModel.PYSUM)
