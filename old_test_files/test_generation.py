from src.model_use import TextGeneration
from src.utils import DEFAULT_DECODING_STRATEGY, MEDIUM
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.torch_loader import GenerationInput
from src.model_training import add_special_tokens

from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    context_input = GenerationInput(P1="Ceci est le début de phrase, ",
                                    P3="et ceci sera la fin de la phrase.",
                                    genre=["horror"],
                                    entities=["Gael", "Alex", "Thomas"],
                                    size=MEDIUM,
                                    summary="Je voudrai parler de ceci")


    print("PREDICTION WITH CONTEXT")

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(
        {'eos_token': '[EOS]',
         'pad_token': '[PAD]',
         'additional_special_tokens': ['[P1]', '[P2]', '[P3]', '[S]', '[M]', '[L]', '[T]', '[Sum]', '[Ent]']}
    )
    model.resize_token_embeddings(len(tokenizer))
    GPT2_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

    text_generator_with_context = TextGeneration(GPT2_model, use_context=True)

    predictions = text_generator_with_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)

    del model, tokenizer, GPT2_model

    print("\n", "-"*100, "\nPREDICTION WITHOUT CONTEXT")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

    text_generator_without_context = TextGeneration(GPT2_model, use_context=False)

    predictions = text_generator_without_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)