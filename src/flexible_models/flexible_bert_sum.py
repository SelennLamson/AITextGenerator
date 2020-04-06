from .flexible_model import FlexibleSummarizer
from src.utils import *

# Summarizers
from eazymind.nlp.eazysum import Summarizer as Sumzer
from summarizer import Summarizer
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
import time

class FlexibleBERTSum(FlexibleSummarizer):
	def __init__(self):
		"""
		Initializes a BERT-SUM model.
		:param min_length: The min length of the summary
		"""
		super().__init__()
		self.bert_sum_model = Summarizer()
		self.bart = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
		self.bart.eval()
		self.bart_tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
		self.key = 'a6bd47ccfdc8f0ffc5b904cd7d90f840'
		self.pgn = Sumzer(self.key)
		self.auto_abstractor = AutoAbstractor()  # Object of automatic summarization.
		self.auto_abstractor.tokenizable_doc = SimpleTokenizer()  # Set tokenizer.
		self.auto_abstractor.delimiter_list = [".", "\n"]  # Set delimiter for making a list of sentence.
		self.abstractable_doc = TopNRankAbstractor()  # Object of abstracting and filtering document.


	def predict(self, inputs: List[str]) -> List[str]:
		"""
		Performs summarization on each paragraph
		:param inputs: list of strings.
		:return: one summary
		"""
		# Bert sum model
		start =time.time()
		outputs = self.bert_sum_model(inputs)  # self.min_length
		result = [''.join(outputs)]
		end = time.time()
		print( 'bert_sum', end - start)

		# BART
		start = time.time()
		bart_inputs = self.bart_tokenizer.batch_encode_plus([inputs], max_length=1024, return_tensors='pt')
		bart_sum_ids = self.bart.generate(bart_inputs['input_ids'], num_beams=4, max_length=40, early_stopping=True)
		bart_result = [self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
		               for g in bart_sum_ids]
		end = time.time()
		print('bart', end - start)

		# Point Generator Network
		start = time.time()
		inputs = inputs.replace('“', '')
		inputs = inputs.replace('”', '')
		pgn_result = self.pgn.run(inputs)
		end = time.time()
		print('PGN', end - start)

		# Pysummarization
		start = time.time()
		result_dict = self.auto_abstractor.summarize(inputs, self.abstractable_doc)
		# Pb index here. Wrong index in tuple in front of score. use enumerate.
		pysum_result = result_dict['summarize_result'][max(result_dict['scoring_data'], key=lambda x: x[1])[0]]
		end = time.time()
		print('pysum', end - start)

		return [result, bart_result, pgn_result, pysum_result]



