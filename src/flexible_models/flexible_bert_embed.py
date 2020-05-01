import numpy as np
import transformers as ppb
import torch
from src.utils import *

from .flexible_model import FlexibleModel


class FlexibleBERTEmbed(FlexibleModel):
    def __init__(self, max_length: int, batch_size: int, cuda_device: int = -2):
        """
        Initializes a BERT embedding model (using [CLS] token).
        :param max_length: The maximum length (in char) the model can handle
        :param batch_size: Batch-size when predicting
        :param cuda_device: Device number cuda should use (-1 = CPU, -2 = AUTO)
        """
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size

        if cuda_device == -2:
            if torch.cuda.is_available():
                self.cuda_device = 0
            else:
                self.cuda_device = -1
        else:
            self.cuda_device = cuda_device

        # Retrieving BERT model and weights handles
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_model = model_class.from_pretrained(pretrained_weights)
        if self.cuda_device != -1:
            self.bert_model.to('cuda:' + str(self.cuda_device))

        self.pad_token = self.tokenizer.pad_token_id
        self.cls_token = self.tokenizer.cls_token_id
        self.sep_token = self.tokenizer.sep_token_id

    def predict(self, inputs: List[str], verbose=1) -> np.ndarray:
        """
        Performs embedding on strings of any length.
        :param inputs: list of N strings.
        :param verbose: 0 for silent, 1 to display progress
        :return: array of shape (N, 768) containing embedding vectors for each input.
        """

        # Encoding without special tokens (will be added manually)
        tokenized_strings = [self.tokenizer.encode(x, add_special_tokens=False) for x in inputs]

        # Splitting at max_length - 2 to keep some space for special tokens
        split_inputs, split_information = token_batch_splitter(tokenized_strings, self.max_length - 2)
        assert all([len(toks) <= self.max_length - 2 for toks in split_inputs])
        assert len(split_inputs) >= len(tokenized_strings)

        # Adding special tokens to every input and padding every input to max_length
        padded = torch.LongTensor(
            [[self.cls_token] + toks + [self.sep_token] + [self.pad_token] * (self.max_length - len(toks) - 2) for toks
             in split_inputs])
        assert padded.shape[1] == self.max_length

        # Building attention mask to hide padding
        ones = torch.ones_like(padded)
        zeros = torch.zeros_like(padded)
        attention_masks = torch.where(padded != self.pad_token, ones, zeros)
        del ones
        del zeros
        assert all([padded.shape[dim] == attention_masks.shape[dim] for dim in range(len(padded.shape))])

        # Saving memory
        del split_inputs
        del tokenized_strings

        # Preparing embeddings output tensor
        embeddings = torch.zeros((len(padded), 768))

        # Transferring to GPU if using it
        if self.cuda_device != -1:
            padded = padded.to('cuda:' + str(self.cuda_device))
            attention_masks = attention_masks.to('cuda:' + str(self.cuda_device))
            embeddings = embeddings.to('cuda:' + str(self.cuda_device))

        # Embedding the sentences with BERT
        start_i = 0
        batch_i = 0
        while start_i < len(padded):
            if verbose:
                print("\rEmbedding {:.2f}%".format(start_i / len(padded) * 100), end="")

            input_ids = padded[start_i:start_i + self.batch_size]
            attention_mask = attention_masks[start_i:start_i + self.batch_size]

            # The embeddings of the [CLS] tokens are the embeddings of the first token of each sentence
            with torch.no_grad():
                embeddings[start_i:start_i + self.batch_size, :] = \
                self.bert_model.forward(input_ids, attention_mask=attention_mask)[0][:, 0, :]

            # Next batch
            start_i += self.batch_size
            batch_i += 1

        # Retrieving the computed values into a numpy array
        if self.cuda_device != -1:
            embeddings = embeddings.cpu()
        embeddings = embeddings.numpy()

        # Merging the values back when input was split before, using mean as a reduce operation
        return np.array(batch_merger(embeddings, split_information, merge_function=lambda x: np.mean(x, axis=0)))
