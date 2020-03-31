from src.model_evaluation import Evaluation
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER

from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""
Script to test the Evaluation module
"""

BERT_NER_BASE = 'models/entity_recognition/bert_ner_base/'
PATH_TO_REPO = 'data/preproc/'

DECODING_STRATEGY = {
    'do_sample': True,
    'max_length': 50,
    'top_k': 50,
    'top_p': 0.95
}

if __name__ == '__main__':
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token

    GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    BERT_NER_model = FlexibleBERTNER(BERT_NER_BASE, batch_size=5, max_length=128)

    evaluation = Evaluation(GPT_model=GPT2_model,
                            GPT_tokenizer=GPT2_tokenizer,
                            BERT_NER_model=BERT_NER_model,
                            path_to_repo=PATH_TO_REPO,
                            batch_size=1,
                            decoding_strategy=DECODING_STRATEGY)

    metrics = evaluation.compute_metrics(nb_examples_to_evaluate=5)
    print(metrics)


