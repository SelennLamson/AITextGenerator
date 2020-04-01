from src.model_evaluation import Evaluation
from src.flexible_models import FlexibleBERTNER, FlexibleGPT2

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
    GPT2_model = FlexibleGPT2(model=GPT2LMHeadModel.from_pretrained('gpt2'),
                              tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
                              decoding_strategy=DECODING_STRATEGY)

    BERT_NER_model = FlexibleBERTNER(BERT_NER_BASE, batch_size=5, max_length=128)

    evaluation = Evaluation(GPT2_model=GPT2_model,
                            BERT_NER_model=BERT_NER_model,
                            path_to_repo=PATH_TO_REPO,
                            batch_size=1)

    metrics = evaluation.compute_metrics(nb_examples_to_evaluate=5)
    print(metrics)


