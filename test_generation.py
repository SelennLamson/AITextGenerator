from src.model_use import TextGeneration
from src.utils import DEFAULT_DECODING_STRATEGY, MEDIUM
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.torch_loader import GenerationInput

from transformers import GPT2LMHeadModel, GPT2Tokenizer

P1 = "Frodo had a very trying time that afternoon. A false rumour that the  \
whole household was being distributed free spread like wildfire; and before \
long the place was packed with people who had no business there, but could \
not be kept out. Fabels got torn off and mixed, and quarrels broke out. Some \
people tried to do swaps and deals in the hall; and others tried to make off \
with minor items not addressed to them, or with anything that seemed \
unwanted or unwatched. The road to the gate was blocked with barrows and \
handcarts."

P3 = "The Sackville-Bagginses were rather offensive. They began by offering \
him bad bargain-prices (as between friends) for various valuable and \
unlabelled things. When Frodo replied that only the things specially \
directed by Bilbo were being given away, they said the whole affair was very \
fishy. "

if __name__ == "__main__":

    context_input = GenerationInput(P1=P1, P3=P3,
                                    genre=["horror"],
                                    persons=["Frodo"],
                                    size=MEDIUM,
                                    summary="hobbits")

    print("PREDICTION WITH CONTEXT WITH SPECIAL TOKENS")
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

    print("\n", "-"*100, "\n", "PREDICTION WITH CONTEXT WITHOUT SPECIAL TOKENS")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

    text_generator_with_context = TextGeneration(GPT2_model, use_context=True)

    predictions = text_generator_with_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)

    del model, tokenizer, GPT2_model

    print("\n", "-"*100, "\nPREDICTION WITHOUT CONTEXT NO SPECIAL TOKENS")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

    text_generator_without_context = TextGeneration(GPT2_model, use_context=False)

    predictions = text_generator_without_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)

    print("\n", "-"*100, "\nPREDICTION WITHOUT CONTEXT WITH SPECIAL TOKENS")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)
    tokenizer.add_special_tokens(
        {'eos_token': '[EOS]',
         'pad_token': '[PAD]',
         'additional_special_tokens': ['[P1]', '[P2]', '[P3]', '[S]', '[M]', '[L]', '[T]', '[Sum]', '[Ent]']}
    )
    model.resize_token_embeddings(len(tokenizer))
    text_generator_without_context = TextGeneration(GPT2_model, use_context=False)

    predictions = text_generator_without_context(context_input, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print('prediction n°', i, ': ', prediction)