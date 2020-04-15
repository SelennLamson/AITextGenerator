from src.model_evaluation.metrics.flexible_metrics import Metrics

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import torch
import numpy as np
import pandas as pd

class SumPerplexity(Metrics):
    """
    Perplexity metrics
    Score the sentences by GPT2 model :
     -> compute perplexity of T5, BART and pysum summaries for GPT2 internal probability distribution
     -> average perplexity between summarizers
    """
    def __init__(self, **kwargs):
        """
        Initialize the GPT2 model (base from huggingface transformers) that will be used to compute the perplexity
        """
        super().__init__()
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if torch.cuda.is_available():
            self.gpt2_model.cuda()
        self.summarizer = kwargs['summarizer']

    def __call__(self, predicted_sentences, original_contexts):
        """
        :param predicted_sentences: list[str] batch of sentences
        :param original_contexts: list[TrainInput] corresponding to original training examples
        :return: pd.DataFrame [perplexity]
        """
        sum_perplexity = self.perplexity(original_contexts)
        return pd.DataFrame(columns=['sum_perplexity'], data=sum_perplexity)

    def perplexity(self, original_contexts):
        """
        Score the sentences by GPT2 model :
         -> perplexity of each summary for GPT2 internal probability distribution
        :param original_contexts: list[TrainInput] corresponding to original training examples
        :return: np.array that contains perplexity score
        """
        perplexity = []

        with torch.no_grad():
            for original_context in original_contexts:
                if self.summarizer == '' or original_context.summaries[self.summarizer] == '':
                    perplexity.append('NaN')
                else:
                    input_ids = self.gpt2_tokenizer.encode(original_context.summaries[self.summarizer], return_tensors='pt')
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                    output = self.gpt2_model.forward(input_ids, labels=input_ids)
                    cross_entropy_loss = output[0].detach().cpu()
                    perplexity.append(math.exp(cross_entropy_loss.item()))

        return np.array(perplexity)
