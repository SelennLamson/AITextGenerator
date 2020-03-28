from torch_loader import VectorizeParagraph, DatasetFromRepo
from transformers import GPT2Tokenizer
import random

"""
Script to test the vectorization module
The idea is to randomly one example from the dataset and then :
1/ vectorize the concataned sentence
2/ print the de-vectorize version and qualitatively check if it is ok
"""

JSON_FILE_PATH = "data/ent_sum/"

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vectorize_paragraph = VectorizeParagraph(tokenizer, block_size=1020)

    tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                  'eos_token': '[EOS]',
                                  'additional_special_tokens': ['[P1]', '[P3]', '[S]', ' [M]',
                                                                '[L]', '[T]', '[Sum]', '[Ent]']})

    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    idx = random.randint(0, len(novels_dataset)-1)
    print(tokenizer.decode(novels_dataset[idx]))


