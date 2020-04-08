from .flexible_model import FlexibleModel
from typing import List
from enum import Enum
from summarizer import Summarizer
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, TFT5ForConditionalGeneration
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from tqdm.notebook import tqdm
import torch

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

        print("Loading model : ", str(summarizer))
        if self.summarizer == SummarizerModel.BERT_SUM:
            self.model = Summarizer()

        if self.summarizer == SummarizerModel.T5:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
            self.decoding_strategy = {}
            if torch.cuda.is_available():
                print("Put T5 on cuda")
                self.model.cuda()
            else:
                print("Warning ! Using T5 wo cuda available")

        if self.summarizer == SummarizerModel.BART:
            self.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
            self.model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
            self.decoding_strategy = {'temperature':1.4, 'num_beams':5, 'min_length':25, 'max_length':65}
            print("Use for decoding strategy :", self.decoding_strategy)
            if torch.cuda.is_available():
                print("Put BART on cuda")
                self.model.cuda()
            else:
                print("Warning ! Using BART wo cuda available")

        if self.summarizer == SummarizerModel.PYSUM:
            self.model = AutoAbstractor()
            self.model.tokenizable_doc = SimpleTokenizer()
            self.model.delimiter_list = ['.','\n']
            self.doc_filtering = TopNRankAbstractor()

    def predict(self, paragraphs: List[str]) -> List[str]:
        """
        Performs summarization on each paragraph
        :param paragraphs: list of strings.
        :return: list[str] : summary for each input
        """
        if self.summarizer == SummarizerModel.BERT_SUM:
            return [''.join(self.model(paragraph, ratio=0.15, max_length=300)) for paragraph in tqdm(paragraphs)]

        if self.summarizer == SummarizerModel.T5 or self.summarizer == SummarizerModel.BART:
            def predict_on_single_batch(batch):
                # batch must be a list of batch_size paragrah (str)
                if self.summarizer == SummarizerModel.T5:
                    inputs_ids = self.tokenizer.batch_encode_plus(batch, return_tensors='tf',
                                                                  max_length=1024, pad_to_max_length=True)
                else:
                    inputs_ids = self.tokenizer.batch_encode_plus(batch, return_tensors='pt',
                                                                  max_length=1024, pad_to_max_length=True)

                inputs_ids = inputs_ids['input_ids'].cuda() if torch.cuda.is_available() else inputs_ids['input_ids']
                outputs = self.model.generate(inputs_ids, **self.decoding_strategy)
                return [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for output in outputs]

            summaries = []
            for i in tqdm(range(len(paragraphs)//self.batch_size)):
                summaries += predict_on_single_batch(paragraphs[i * self.batch_size: (i+1) * self.batch_size])
            if len(paragraphs) % self.batch_size != 0:
                summaries += predict_on_single_batch(paragraphs[len(paragraphs)//self.batch_size * self.batch_size:])

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

            return [one_paragraph_summarization(paragraph) for paragraph in tqdm(paragraphs)]








