from src.torch_loader import VectorizeParagraph, DatasetFromRepo, VectorizeMode
from src.model_training import add_special_tokens
from src.model_use import GenerationInput
from src.utils import MEDIUM

import random
from transformers import GPT2Tokenizer

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
    idx = random.randint(0, len(DatasetFromRepo(path=JSON_FILE_PATH))-1)

    print("--- TEST VECTORIZER IN TRAIN MODE WITH FULL CONTEXT ---")
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.TRAIN, use_context=True)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    print(tokenizer.decode(novels_dataset[idx]))

    print("--- TEST VECTORIZER IN TRAIN MODE WITHOUT FULL CONTEXT ---")
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.TRAIN, use_context=False)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    idx = random.randint(0, len(novels_dataset)-1)
    print(tokenizer.decode(novels_dataset[idx]))

    print("\n", '-' * 100, "\n")
    print("--- TEST VECTORIZER IN EVAL MODE WITH FULL CONTEXT ---")
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.EVAL, use_context=True)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    idx = random.randint(0, len(novels_dataset)-1)
    tokenize_context, P2, P3 = novels_dataset[idx]
    print(tokenizer.decode(tokenize_context))

    print("\n", '-'*100,"\n")
    print("--- TEST VECTORIZER IN EVAL MODE WITHOUT FULL CONTEXT ---")
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.EVAL, use_context=False)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)
    tokenize_context, P2, P3 = novels_dataset[idx]
    print(tokenizer.decode(tokenize_context))

    context_input = GenerationInput(P1="Ceci est le début de phrase, ",
                                    P3="et ceci sera la fin de la phrase.",
                                    genre=["horror"],
                                    entities=["Gael", "Alex", "Thomas"],
                                    size=MEDIUM,
                                    context="Je voudrai parler de ceci")

    print("\n", '-'*100,"\n")
    print("--- TEST VECTORIZER IN GENERATE MODE WITH FULL CONTEXT ---")
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.GENERATE, use_context=True)
    tokenize_context = vectorize_paragraph(context_input.vectorizer_input_format())
    print(tokenizer.decode(tokenize_context))

    print("\n", '-'*100,"\n")
    print("--- TEST VECTORIZER IN GENERATE MODE WITHOUT FULL CONTEXT ---")
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.GENERATE, use_context=False)
    tokenize_context = vectorize_paragraph(context_input.vectorizer_input_format())
    print(tokenizer.decode(tokenize_context))
