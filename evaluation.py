from src.model_evaluation import GPT2EvaluationScript
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.utils import DEFAULT_DECODING_STRATEGY
from datetime import datetime
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk

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

    parser.add_argument("--batch_size", default=8,
                        help="Batch size that will be used by all models, by default 8")

    args = parser.parse_args()

    print("Loading the GPT2 fine-tuned model ...")
    gpt_2 = FlexibleGPT2(model=GPT2LMHeadModel.from_pretrained(args.model),
                         tokenizer=GPT2Tokenizer.from_pretrained(args.model),
                         decoding_strategy=DEFAULT_DECODING_STRATEGY)

    print("Evaluating the model ...")
    script = GPT2EvaluationScript(path_to_data_folder=args.data,
                                  batch_size=args.batch_size,
                                  path_to_bert_ner=args.ner)

    moment = str(datetime.now().strftime("%m_%d-%H_%M"))
    script(generations_path=args.output + 'generation' + moment + '.json',
           results_path=args.output + 'metrics.json' + moment + '.json',
           GPT2_model=gpt_2,
           compute_bert_similarity=True,
           compute_entites_iou=True,
           compute_gpt2_perplexity=True,
           compute_residual_tokens=True,
           verbose=1)
