from src.model_evaluation import GPT2EvaluationScript
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.utils import DEFAULT_DECODING_STRATEGY, ALL_METRICS
from datetime import datetime
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
import os

"""
Script to evaluate one model
"""

if __name__ == '__main__':
    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the folder containing the novels on which the model will be evaluated")

    parser.add_argument("--output", default="", type=str,
                        help="Path to the folder where the results with be stored, by default current folder")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to the folder containing the fine-tune models that need to be evaluated")

    parser.add_argument("--ner",  required=True,
                        help="Path to the folder containing the weights of BERT NER model")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size that will be used by all models, by default 8")

    parser.add_argument("--sum", type=str, default="",
                        help="Which summary to use for text generation : KW, T5, BART, PYSUM. \
                              by default do not use any summaries")

    parser.add_argument("--name", type=str, default="", help="Name of save, by default the time")

    args = parser.parse_args()

    print("Loading the GPT2 fine-tuned model ...")
    gpt_2 = FlexibleGPT2(model=GPT2LMHeadModel.from_pretrained(args.model),
                         tokenizer=GPT2Tokenizer.from_pretrained(args.model),
                         decoding_strategy=DEFAULT_DECODING_STRATEGY)

    print("Evaluating the model ...")
    script = GPT2EvaluationScript(path_to_data_folder=args.data,
                                  batch_size=args.batch_size,
                                  path_to_bert_ner=args.ner,
                                  summarizer=args.sum)

    save_name = str(datetime.now().strftime("%d_%b_%Hh%M")) if args.name == "" else args.name
    generation_path = args.output + 'generation_' + save_name + '.json'
    results_path = args.output + 'metrics_' + save_name + '.csv'

    if not os.path.exists(generation_path):
        script.generate_texts(generation_path, gpt_2, verbose=1)

    script.compute_metrics(generation_path, results_path, ALL_METRICS, verbose=1)

