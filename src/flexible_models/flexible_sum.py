from .flexible_model import FlexibleModel
from typing import List
from enum import Enum
from summarizer import Summarizer
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, TFT5ForConditionalGeneration
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from tqdm import tqdm

class SummarizerModel(Enum):
    T5 = 0
    BART = 1
    BERT_SUM = 2
    PYSUM = 3

    def __str__(self):
        if self == SummarizerModel.T5:
            return 'T5'
        if self == SummarizerModel.BART:
            return 'BART'
        if self == SummarizerModel.BERT_SUM:
            return 'BERT_SUM'
        if self == SummarizerModel.PYSUM:
            return 'PYSUM'

class FlexibleSum(FlexibleModel):
    """
    FlexibleSum class allows the use of 4 differents type of summarizers
    - T5
    - BART
    - BERT_SUM
    - PYSUM
    """
    def __init__(self, summarizer, batch_size=1):
        """
        :param summarizer: SummarizerModel value
        :param batch_size : [int] batch size for summarizer input (for T5 and BART)
        """
        super().__init__()
        self.summarizer = summarizer
        self.batch_size = batch_size
        if self.summarizer == SummarizerModel.BERT_SUM:
            self.model = Summarizer()

        if self.summarizer == SummarizerModel.T5:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
            self.decoding_strategy = {}

        if self.summarizer == SummarizerModel.BART:
            self.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
            self.model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
            self.decoding_strategy = {'num_beams':2, 'max_length':40, 'early_stopping':True}

        if self.summarizer == SummarizerModel.PYSUM:
            self.model = AutoAbstractor()
            self.model.tokenizer = SimpleTokenizer()
            self.model.delimiter_list = ['.','\n']
            self.doc_filtering = TopNRankAbstractor()

    def predict(self, paragraphs: List[str]) -> List[str]:
        """
        Performs summarization on each paragraph
        :param paragraphs: list of strings.
        :return: list[str] : summary for each input
        """
        if self.summarizer == SummarizerModel.BERT_SUM:
            return [''.join(self.model(paragraph, ratio=0.15, max_length=300)) for paragraph in paragraphs]

        if self.summarizer == SummarizerModel.T5 or self.summarizer == SummarizerModel.BART:
            def predict_on_single_batch(batch):
                # batch must be a list of batch_size paragrah (str)
                if self.summarizer == SummarizerModel.T5:
                    inputs_ids = self.tokenizer.batch_encode_plus(batch, return_tensors='tf', max_length=1024)
                else:
                    inputs_ids = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', max_length=1024)
                outputs = self.model.generate(inputs_ids['input_ids'], **self.decoding_strategy)
                return [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for output in outputs]

            summaries = []
            i = 0
            while i + self.batch_size < len(paragraphs):
                summaries += predict_on_single_batch(paragraphs[i:i+self.batch_size])
            summaries += predict_on_single_batch(paragraphs[i:])
            return summaries

        if self.summarizer == SummarizerModel.PYSUM:
            def one_paragraph_summarization(single_paragraph):
                result_dict = self.model.summarize(single_paragraph, self.doc_filtering)
                max = 0
                for i, item in enumerate(result_dict['scoring_data']):
                    if item[1] > max:
                        id = i
                        max = item[1]
                pysum_result = ''.join(result_dict['summarize_result'][id])
                return pysum_result.replace('\n', '')

            return [one_paragraph_summarization(paragraph) for paragraph in paragraphs]








