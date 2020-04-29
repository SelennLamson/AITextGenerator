from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

import matplotlib.pyplot as plt

class GPT2LMSegmentModel(GPT2LMHeadModel):
	def __init__(self, config):
		super().__init__(config)
		self.min_special = None
		self.p2_token = None
		self.max_special = None
		self.eos_token = None
		self.eos_weight = 10

	def set_special_tokens(self, tokenizer: GPT2Tokenizer):
		self.p2_token = self.min_special = tokenizer.bos_token_id
		self.eos_token = tokenizer.eos_token_id
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

	# def forward(
	# 		self,
	# 		input_ids=None,
	# 		past=None,
	# 		attention_mask=None,
	# 		token_type_ids=None,
	# 		position_ids=None,
	# 		head_mask=None,
	# 		inputs_embeds=None,
	# 		labels=None,
	# ):
	# 	r"""
	# 	labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
	# 		Labels for language modeling.
	# 		Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
	# 		Indices are selected in ``[-100, 0, ..., config.vocab_size]``
	# 		All labels set to ``-100`` are ignored (masked), the loss is only
	# 		computed for labels in ``[0, ..., config.vocab_size]``
	#
	# Return:
	# 	:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
	# 	loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
	# 		Language modeling loss.
	# 	prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
	# 		Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
	# 	past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
	# 		Contains pre-computed hidden-states (key and values in the attention blocks).
	# 		Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
	# 		should not be passed as input ids as they have already been computed.
	# 	hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
	# 		Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
	# 		of shape :obj:`(batch_size, sequence_length, hidden_size)`.
	#
	# 		Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	# 	attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
	# 		Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
	# 		:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
	#
	# 		Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	# 		heads.
	#
	# Examples::
	#
	# 	import torch
	# 	from transformers import GPT2Tokenizer, GPT2LMHeadModel
	#
	# 	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	# 	model = GPT2LMHeadModel.from_pretrained('gpt2')
	#
	# 	input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
	# 	outputs = model(input_ids, labels=input_ids)
	# 	loss, logits = outputs[:2]
	#
	# 	"""
	# 	transformer_outputs = self.transformer(
	# 		input_ids,
	# 		past=past,
	# 		attention_mask=attention_mask,
	# 		token_type_ids=token_type_ids,
	# 		position_ids=position_ids,
	# 		head_mask=head_mask,
	# 		inputs_embeds=inputs_embeds,
	# 	)
	# 	hidden_states = transformer_outputs[0]
	#
	# 	lm_logits = self.lm_head(hidden_states)
	#
	# 	outputs = (lm_logits,) + transformer_outputs[1:]
	# 	if labels is not None:
	# 		# Shift so that tokens < n predict n
	# 		shift_logits = lm_logits[..., :-1, :].contiguous()
	# 		shift_labels = labels[..., 1:].contiguous()
	#
	# 		eos_position = torch.abs(shift_labels - self.eos_token).argmin(dim=1)
	# 		token_position = torch.arange(0, shift_labels.shape[1]).unsqueeze(0).repeat([eos_position.shape[0], 1])
	# 		eos_position = eos_position.unsqueeze(1).repeat([1, token_position.shape[1]])
	#
	# 		max_dist = 20
	# 		eos_distance = (1 - torch.abs(token_position - eos_position).double() / max_dist).clamp(0, 1)
	#
	# 		nll_loss_token = torch.nn.NLLLoss()
	#
	# 		log_softmax = torch.log_softmax(shift_logits.view(-1, shift_logits.size(-1)), 1)
	# 		nll_token = nll_loss_token(log_softmax, shift_labels.view(-1))
	# 		nll_eos = torch.mean(torch.exp(log_softmax[:, self.eos_token]) * ((1 - eos_distance)**2).view(-1))
	#
	# 		loss = self.eos_weight * nll_eos + nll_token
	#
	# 		outputs = (loss,) + outputs + (nll_token, nll_eos)
	# 	return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
