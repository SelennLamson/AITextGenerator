from src.model_evaluation.metrics.flexible_metrics import Metrics

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import torch
import numpy as np
import pandas as pd

class GPT2Perplexity(Metrics):
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
        :param sentences: list[str] batch of sentences
        :param gpt2_model: [FlexibleGPT2] pre-train GPT2 model encapsulate in a FlexibleGPT2 object
        :return: np.array that contains perplexity score
        """
        perplexity = []

        with torch.no_grad():
            for original_context in original_contexts:
                input_ids_bart = self.gpt2_tokenizer.encode(original_context.summaries['BART'], return_tensors='pt')
                input_ids_pysum = self.gpt2_tokenizer.encode(original_context.summaries['PYSUM'], return_tensors='pt')
                input_ids_t5 = self.gpt2_tokenizer.encode(original_context.summaries['T5'], return_tensors='pt')
                if torch.cuda.is_available():
                    input_ids_bart = input_ids_bart.cuda()
                    input_ids_pysum = input_ids_pysum.cuda()
                    input_ids_t5 = input_ids_t5.cuda()
                output_bart = self.gpt2_model.forward(input_ids_bart, labels=input_ids_bart)
                output_pysum = self.gpt2_model.forward(input_ids_pysum, labels=input_ids_pysum)
                output_t5 = self.gpt2_model.forward(input_ids_t5, labels=input_ids_t5)
                cross_entropy_loss_bart = output_bart[0].detach().cpu()
                cross_entropy_loss_pysum = output_pysum[0].detach().cpu()
                cross_entropy_loss_t5 = output_t5[0].detach().cpu()
                cross_entropy_loss = (cross_entropy_loss_t5 + cross_entropy_loss_pysum + cross_entropy_loss_bart) / 3
                perplexity.append(math.exp(cross_entropy_loss.item()))

        return np.array(perplexity)
