from src.model_use import TextGeneration, GenerationInput
from src.utils import DEFAULT_DECODING_STRATEGY
from src.flexible_models import FlexibleGPT2

from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    GPT2_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

    text_generator = TextGeneration(GPT2_model)

    context_input = GenerationInput(P1="Ceci est le début de phrase, ",
                                    P3="et ceci sera la fin de la phrase.",
                                    genre=["horror"],
                                    entities=["Gael", "Alex", "Thomas"],
                                    size="M",
                                    context="Je voudrai parler de ceci")

    # To check if the context is correctly Vectorize
    predictions = text_generator(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)
