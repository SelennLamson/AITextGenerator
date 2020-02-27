from dataset_loader.torch_dataset_generation import DatasetFromJsonFile, DatasetFromJsonRepo
from dataset_loader.GPT_2_tokenization import TransformParagraphes
import json
from transformers import GPT2Tokenizer

JSON_FILE_PATH = "../data/ent_sum/"

if __name__ == "__main__":
    transform = TransformParagraphes()

    one_novel_dataset = DatasetFromJsonRepo(path=JSON_FILE_PATH, transform=transform)
    print("nb de paragraph", len(one_novel_dataset))
    print("premier triplet", one_novel_dataset[0])
