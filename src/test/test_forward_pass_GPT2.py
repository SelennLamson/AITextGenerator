from torch_loader import DatasetFromRepo, VectorizeParagraph
from model_training import add_special_tokens
from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""
Script to test one forward pass in GPT2 with example from our custom dataset loader 
"""

JSON_FILE_PATH = "data/ent_sum/"

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    add_special_tokens(model, tokenizer)

    vectorize_paragraph = VectorizeParagraph(tokenizer, block_size=1020)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)

    output = model(novels_dataset[4])
    print(output[0].shape)

