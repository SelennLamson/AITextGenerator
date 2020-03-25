
##########################################
# GAEL's TEST FILE, PLEASE DO NOT MODIFY #
##########################################

from src.torch_loader import DatasetFromRepo
from src.torch_loader import VectorizeParagraph
from transformers import GPT2LMHeadModel


JSON_FILE_PATH = "data/ent_sum/"

if __name__ == "__main__":
    vectorize_paragraph = VectorizeParagraph(block_size=1020)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    print(novels_dataset[0].shape)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(vectorize_paragraph.tokenizer))
    output = model(novels_dataset[0])

    print(output[0].shape)