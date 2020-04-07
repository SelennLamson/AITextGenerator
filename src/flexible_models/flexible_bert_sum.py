from .flexible_model import FlexibleSummarizer
from src.utils import *

# from eazymind.nlp.eazysum import Summarizer as Sumzer
from summarizer import Summarizer
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, TFT5ForConditionalGeneration
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
import random


class FlexibleBERTSum(FlexibleSummarizer):
	def __init__(self):
		"""
		Initializes a BERT-SUM model.
		:param min_length: The min length of the summary
		"""
		super().__init__()
		self.bert_sum_model = Summarizer()
		self.bart_tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
		self.bart = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
		self.bart.eval()
		self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
		self.t5 = TFT5ForConditionalGeneration.from_pretrained('t5-small')
		self.auto_abstractor = AutoAbstractor()  # Object of automatic summarization.
		self.auto_abstractor.tokenizable_doc = SimpleTokenizer()  # Set tokenizer.
		self.auto_abstractor.delimiter_list = ['.','\n']  # Set delimiter for making a list of sentence.
		self.abstractable_doc = TopNRankAbstractor()  # Object of abstracting and filtering document.
		# self.key = 'a6bd47ccfdc8f0ffc5b904cd7d90f840'
		# self.pgn = Sumzer(self.key)

	def predict(self, inputs: List[str]) -> List[str]:
		"""
		Performs summarization on each paragraph
		:param inputs: list of strings.
		:return: one summary
		"""
		# Use a random summarizer among those
		n = random.randint(1, 4)

		# Point Generator Network
		"""
		# inputs = inputs.replace('“', '').replace('”', '').replace('’', '')
		try:
			inputs = inputs.encode('Latin-1', 'ignore')
			inputs = inputs.decode('utf-8', 'ignore')
			pgn_result = [self.pgn.run(inputs)]
		except TypeError:
			pgn_result = []
		"""

		if n == 1:
			# Bert sum model
			outputs = self.bert_sum_model(inputs, ratio=0.15, max_length=300)  # self.min_length
			bert_sum_result = [''.join(outputs)]
			return bert_sum_result

		elif n==2:
			# BART
			bart_inputs = self.bart_tokenizer.batch_encode_plus([inputs], max_length=1024, return_tensors='pt')
			bart_sum_ids = self.bart.generate(bart_inputs['input_ids'], num_beams=2, max_length=40, early_stopping=True)
			bart_result = [self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
			               for g in bart_sum_ids]
			return bart_result

		elif n==3:
			# T5
			t5_inputs = self.t5_tokenizer.encode(inputs, return_tensors='tf')
			t5_sum_ids = self.t5.generate(t5_inputs)
			t5_result = [self.t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
			               for g in t5_sum_ids]
			return t5_result

		else:
			# Pysummarization
			result_dict = self.auto_abstractor.summarize(inputs, self.abstractable_doc)
			max = 0
			for i, item in enumerate(result_dict['scoring_data']):
				if item[1] > max:
					id = i
					max = item[1]
			pysum_result = ''.join(result_dict['summarize_result'][id])
			pysum_result = [pysum_result.replace('\n', '')]
			return pysum_result





