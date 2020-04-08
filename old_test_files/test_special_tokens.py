from transformers import GPT2Tokenizer
from src.model_training import add_special_tokens
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print('All special tokens:', tokenizer.all_special_tokens)
print('All special ids:', tokenizer.all_special_ids)

print("\n Add special tokens \n")
tokenizer.add_special_tokens(
    {'bos_token': '[P2]',
     'additional_special_tokens': ['[P1]', '[P3]', '[S]', '[M]', '[L]', '[T]', '[Sum]', '[Ent]']}
)

print('All special tokens:', tokenizer.all_special_tokens)
print('All special ids:', tokenizer.all_special_ids)
print("EOS token _ids:", tokenizer.eos_token_id)