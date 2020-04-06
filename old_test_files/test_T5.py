from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.utils import *
from src.model_use import GenerationInput, TextGeneration
from src.flexible_models.flexible_LG import FlexibleLG

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

tokenizer.add_special_tokens({'bos_token': '[P2]'})

T5_model = FlexibleLG(model, tokenizer, DEFAULT_DECODING_STRATEGY)

context_input = GenerationInput(P1="Thomas is playing tennis with [P2]",
                                P3="et ceci sera la fin de la phrase.",
                                genre=["horror"],
                                entities=["Gael", "Alex", "Thomas"],
                                size=MEDIUM,
                                context="Je voudrai parler de ceci")

text_generator = TextGeneration(T5_model)

# To check if the context is correctly Vectorize
predictions = text_generator(context_input, nb_samples=3)
for i, prediction in enumerate(predictions):
    print('prediction n°', i, ': ', prediction)

