from src.model_use import TextGeneration
from src.utils import DEFAULT_DECODING_STRATEGY, MEDIUM
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.torch_loader import GenerationInput

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


######################################
# Classic gpt2
######################################

# Define model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define input sequence (P1)
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_p=0.9,
	temperature=1
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


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


input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

######################################
# Our model
######################################
MODEL_PATH = 'data_output/outputs_epoch_3/'
DATA = 'data/Preproc_KW/'

context_input = GenerationInput(P1=P1, P3=P3,
                                    genre=["horror"],
                                    persons=["Frodo"],
                                    size=MEDIUM,
                                    summary="hobbits")
