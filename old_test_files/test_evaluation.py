from src.model_evaluation import GPT2EvaluationScript
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.utils import BERT_NER_BASE, DEFAULT_DECODING_STRATEGY, PREPROC_PATH
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import os
"""
Script to test the Evaluation module
"""

EVAL_PATH = 'data/eval/'
GENERATION_PATH = 'data/outputs/generations.json'
RESULT_PATH = 'data/outputs/results.csv'

if __name__ == '__main__':
    flexible_gpt2_to_evaluate = FlexibleGPT2(
        model=GPT2LMHeadModel.from_pretrained('gpt2'),
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
        decoding_strategy=DEFAULT_DECODING_STRATEGY
    )

    evaluation_script = GPT2EvaluationScript(path_to_data_folder=EVAL_PATH,path_to_bert_ner=BERT_NER_BASE)
    evaluation_script(generations_path=GENERATION_PATH,
                      results_path=RESULT_PATH,
                      GPT2_model=flexible_gpt2_to_evaluate,
                      compute_bert_relationship=True,
                      compute_entities_count=True,
                      compute_bert_similarity=True,
                      compute_gpt2_perplexity=True)
