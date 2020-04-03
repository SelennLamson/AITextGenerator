from src.flexible_models import FlexibleGPT2
from src.utils import DEFAULT_DECODING_STRATEGY
from src.model_evaluation.metrics import gpt2_perplexity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == '__main__':

    GPT2_model = FlexibleGPT2(
        model=GPT2LMHeadModel.from_pretrained('gpt2'),
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
        decoding_strategy=DEFAULT_DECODING_STRATEGY
    )

    input_sentences = [
        "The dog plays with the ball",
        "The ball plays with the dog",
        "Alex is eating in the kitchen",
        "The kitchen is eating Alex"
    ]

    perplexity_list = gpt2_perplexity(input_sentences, GPT2_model)

    for sentence, perplexity in zip(input_sentences, perplexity_list):
        print("Perplexity of sentence: ", sentence, " -> ", perplexity)

