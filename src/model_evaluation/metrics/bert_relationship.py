import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForNextSentencePrediction

from src.model_evaluation.metrics.flexible_metrics import Metrics


class BertRelationship(Metrics):
    """
    1. Compute the probability (for pre-trained BERT) that P3 follow pred P2
    2. Normalize by the probability that P3 follow true P2
    """
    def __init__(self, **kwargs):
        """
        Initialized the BERT model
        :param batch_size: [int] batch size to used for bert
        """
        super().__init__()
        self.batch_size = kwargs['batch_size']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.model.eval()
        #if torch.cuda.is_available():
        #    self.model.cuda()

    def __call__(self, predicted_sentences, original_contexts):
        """
        :param predicted_sentences: list[str] batch of sentences corresponding to the generated P2
        :param original_contexts: list[TrainInput] corresponding to original training examples
        :return: pd.DataFrame["relationship"]
        """
        data = self.bert_relationship(predicted_sentences,
                                      [original_context.P2 for original_context in original_contexts])
        return pd.DataFrame(columns=["relationship"], data=data)

    def bert_relationship_single_batch(self, list_seq_1, list_seq_2):
        """
        Will compute bert relationship of each sentences simultanously
        :param list_seq_1: List[str] list of first sequences
        :param list_seq_2: List[str] list of second sequence
        :return: np.array for each idx : probability that list_seq_2[idx] is the continuation of list_seq_1[idx]
        """
        encoded_dicts = [self.tokenizer.encode_plus(seq_1, seq_2) for (seq_1, seq_2) in zip(list_seq_1, list_seq_2)]
        input_ids = [torch.tensor(encoded_dict['input_ids']) for encoded_dict in encoded_dicts]
        token_type_ids = [torch.tensor(encoded_dict['token_type_ids'], dtype=torch.long) for encoded_dict in encoded_dicts]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        mask = (input_ids != self.tokenizer.pad_token_id).long()

        """
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            mask = mask.cuda()
        """
        ouptput_bert = self.model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        return torch.nn.functional.softmax(ouptput_bert[0], dim=1)[:,0].detach()  #.cpu().numpy()

    def bert_relationship(self, list_seq_1, list_seq_2):
        """
        Will compute bert relationship of each sentences simultanously by batch of batch_size
        :param list_seq_1: List[str] list of first sequences
        :param list_seq_2: List[str] list of second sequence
        :return: np.array for each idx : probability that list_seq_2[idx] is the continuation of list_seq_1[idx]
        """
        number_seq = len(list_seq_1)
        outputs = []
        batch_size = self.batch_size
        i = 0
        while i + batch_size < number_seq:
            outputs.append(self.bert_relationship_single_batch(list_seq_1[i:i+batch_size], list_seq_2[i:i+batch_size]))
            i += batch_size

        outputs.append(self.bert_relationship_single_batch(list_seq_1[i:], list_seq_2[i:]))
        return np.hstack(outputs)

