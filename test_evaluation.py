from src.model_evaluation import metrics, GPT2EvaluationScript
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

    print(hasattr(metrics, 'EntitiesCount'))
    if not os.path.exists(GENERATION_PATH):
        flexible_gpt2_to_evaluate = FlexibleGPT2(
            model=GPT2LMHeadModel.from_pretrained('gpt2'),
            tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
            decoding_strategy=DEFAULT_DECODING_STRATEGY
        )

    evaluation_script = GPT2EvaluationScript(path_to_data_folder=EVAL_PATH,path_to_bert_ner=BERT_NER_BASE)

    if not os.path.exists(GENERATION_PATH):
        evaluation_script.generate_texts(GENERATION_PATH, flexible_gpt2_to_evaluate)

    evaluation_script.compute_metrics(generations_path=GENERATION_PATH,
                                      results_path=RESULT_PATH,
                                      metric_names=["EntitiesCount", "BertSimilarity",
                                                    "BertRelationship", "GPT2Perplexity"])
