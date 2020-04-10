from src.model_evaluation import GPT2EvaluationScript
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.utils import BERT_NER_BASE, DEFAULT_DECODING_STRATEGY, PREPROC_PATH
from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""
Script to test the Evaluation module
"""

if __name__ == '__main__':
    GPT2_model = FlexibleGPT2(
        model=GPT2LMHeadModel.from_pretrained('gpt2'),
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
        decoding_strategy=DEFAULT_DECODING_STRATEGY
    )

    evaluation_script = GPT2EvaluationScript(
        file_ids=["1"],
        path_to_data_folder=PREPROC_PATH,
        path_to_bert_ner=BERT_NER_BASE
    )

    evaluation_script.generate_texts('data/outputs/generations.json', GPT2_model)

