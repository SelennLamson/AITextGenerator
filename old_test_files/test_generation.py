from src.model_use import TextGeneration, GenerationInput
from src.utils import DEFAULT_DECODING_STRATEGY, MEDIUM
from src.flexible_models import FlexibleGPT2

from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    GPT2_model = FlexibleGPT2(model=GPT2LMHeadModel.from_pretrained('gpt2'),
                              tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
                              decoding_strategy=DEFAULT_DECODING_STRATEGY)

    text_generator_with_context = TextGeneration(GPT2_model, use_context=True)
    text_generator_without_context = TextGeneration(GPT2_model, use_context=False)

    context_input = GenerationInput(P1="Ceci est le début de phrase, ",
                                    P3="et ceci sera la fin de la phrase.",
                                    genre=["horror"],
                                    entities=["Gael", "Alex", "Thomas"],
                                    size=MEDIUM,
                                    context="Je voudrai parler de ceci")

    print("PREDICTION WITH CONTEXT")
    predictions = text_generator_with_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)

    print("\n", "-"*100, "\nPREDICTION WITHOUT CONTEXT")
    predictions = text_generator_without_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)