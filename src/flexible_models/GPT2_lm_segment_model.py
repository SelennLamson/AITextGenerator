from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPT2LMSegmentModel(GPT2LMHeadModel):
	def __init__(self, config):
		super().__init__(config)
		self.min_special = None
		self.p2_token = None
		self.max_special = None

	def set_special_tokens(self, tokenizer: GPT2Tokenizer):
		self.p2_token = self.min_special = tokenizer.bos_token_id
		self.max_special = max(tokenizer.additional_special_tokens_ids)

	def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
		assert self.p2_token is not None, "set_special_tokens(tokenizer: GPT2Tokenizer) should be called once when initializing a GPT2LMSegmentModel."

		# only last token for inputs_ids if past is defined in kwargs

		if past:
			input_ids = input_ids[:, -1].unsqueeze(-1)
			type_ids = torch.ones_like(input_ids) * self.p2_token

		else:
			type_ids = torch.zeros_like(input_ids)
			for si in range(input_ids.shape[0]):
				current_special_token = self.max_special
				for ti in range(input_ids.shape[1]):
					curr = input_ids[si, ti]
					if self.min_special <= curr <= self.max_special:
						current_special_token = curr
					type_ids[si, ti] = current_special_token

		return {"input_ids": input_ids, "past": past, "token_type_ids": type_ids}
