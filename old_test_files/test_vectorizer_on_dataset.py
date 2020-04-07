from src.torch_loader import VectorizeParagraph, DatasetFromRepo, VectorizeMode
from src.model_training import add_special_tokens

from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

JSON_FILE_PATH = "data/preproc/"

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    add_special_tokens(tokenizer=tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    vectorize_paragraph = VectorizeParagraph(tokenizer, mode=VectorizeMode.TRAIN, use_context=True)
    novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=vectorize_paragraph)

    def collate(examples):
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_dataloader = DataLoader(novels_dataset,
                                  sampler=RandomSampler(novels_dataset),
                                  batch_size=1,
                                  collate_fn=collate)

    size_list = []
    for i, inputs_ids in tqdm((enumerate(train_dataloader))):
        size_list.append(inputs_ids.shape[1])
        if i == 5000:
            break
    print('max size: ', max(size_list))