from typing import List
import numpy as np
import transformers as ppb
import torch
from src.utils import *

from .flexible_model import FlexibleModel

class FlexibleBERTEmbed(FlexibleModel):
	def __init__(self, max_length:int, batch_size:int, cuda_device:int = -2):
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

	def predict(self, inputs: List[str]) -> np.ndarray:
		"""
		Performs embedding on strings of any length.
		:param inputs: list of N strings.
		:return: array of shape (N, 768) containing embedding vectors for each input.
		"""

		split_strings, split_information = text_batch_splitter(inputs, self.max_length)

		# Preparing sequences in batch
		tokenized = [self.tokenizer.encode(x, add_special_tokens=True) for x in split_strings]
		max_len = max(len(t) for t in tokenized)
		padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

		# Building attention mask to hide padding
		attention_mask_np = np.where(padded != 0, 1, 0)

		# Embedding the sentences with BERT
		embeddings = np.zeros((len(padded), 768))
		start_i = 0
		batch_i = 0
		while start_i < len(padded):
			input_ids = torch.LongTensor(padded[start_i:start_i + self.batch_size])
			attention_mask = torch.tensor(attention_mask_np[start_i:start_i + self.batch_size])

			if self.cuda_device != -1:
				input_ids = input_ids.to('cuda:' + str(self.cuda_device))
				attention_mask = attention_mask.to('cuda:' + str(self.cuda_device))

			with torch.no_grad():
				last_hidden_states = self.bert_model.forward(input_ids, attention_mask=attention_mask)[0][:, 0, :]

			if self.cuda_device != -1:
				last_hidden_states = last_hidden_states.cpu()

			embeddings[start_i:start_i + self.batch_size, :] = last_hidden_states.numpy()
			start_i += self.batch_size
			batch_i += 1

		return np.array(batch_merger(embeddings, split_information, merge_function=lambda x: np.mean(x, axis=0)))


