from src.torch_loader import VectorizeParagraph, DatasetFromRepo, VectorizeMode, GenerationInput, TrainInput
from src.model_training import add_special_tokens
from src.utils import summary_selector
from transformers import GPT2Tokenizer
import argparse

JSON_FILE_PATH = "data/preproc_with_sum/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum", default=[], nargs='+', help="Choose list of summarizers to use from \
                                                                  T5, BART, KW, PYSUM \
                                                                  by default do not use any summariers")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    add_special_tokens(tokenizer=tokenizer)

    summary_to_select = summary_selector(args.sum)
    vectorize_paragraph = VectorizeParagraph(tokenizer=tokenizer,
                                             mode=VectorizeMode.TRAIN,
                                             use_context=True,
                                             select_summary=summary_to_select)

    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    tokenize_context = novels_dataset[0]
    print("CONTEXT SIZE (IN NUMBER TOKENS) : ", len(tokenize_context))
    print(tokenizer.decode(tokenize_context))
