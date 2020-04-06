from src.torch_loader import VectorizeParagraph, DatasetFromRepo
from src.model_training import add_special_tokens
from transformers import GPT2Tokenizer
import random

"""
Script to test the vectorization module in train and evaluation mode 
The idea is to randomly take one example from the dataset and then :
1/ vectorize the concataned sentence
2/ print the de-vectorize version and qualitatively check if it is ok
"""

JSON_FILE_PATH = "data/preproc/"

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    add_special_tokens(tokenizer=tokenizer)

    # TEST VECTORIZER IN TRAIN MODE
    vectorize_paragraph = VectorizeParagraph(tokenizer, block_size=1020, mode='train')
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    idx = random.randint(0, len(novels_dataset)-1)
    print(novels_dataset[idx])
    print(tokenizer.decode(novels_dataset[idx]))

    print("\n", '-'*100,"\n")
    # TEST VECTORIZER IN EVAL MODE
    vectorize_paragraph = VectorizeParagraph(tokenizer, block_size=1020, mode='eval')
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    tokenize_context, P2, _ = novels_dataset[idx]
    print(tokenizer.decode(tokenize_context), " , ", P2)

