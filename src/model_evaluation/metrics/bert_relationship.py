import torch
from torch.nn.utils.rnn import pad_sequence

def bert_relationship(list_seq_1, list_seq_2, BERT_model, BERT_tokenizer):
    """
    :param list_seq_1: List[str] list of first sequences
    :param list_seq_2: List[str] list of second sequence
    :param BERT_model: transformers.BertForNextSentencePrediction
    :param BERT_tokenizer: transformers.BertTokenizer
    :return: np.array for each idx : probability that list_seq_2[idx] is the continuation of list_seq_1[idx]
    """

    encoded_dicts = [BERT_tokenizer.encode_plus(seq_1, seq_2) for (seq_1, seq_2) in zip(list_seq_1, list_seq_2)]
    input_ids = [torch.tensor(encoded_dict['input_ids']) for encoded_dict in encoded_dicts]
    token_type_ids = [torch.tensor(encoded_dict['token_type_ids'], dtype=torch.long) for encoded_dict in encoded_dicts]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=BERT_tokenizer.pad_token_id)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
    mask = (input_ids != BERT_tokenizer.pad_token_id).long()

    BERT_model.eval()
    ouptput_bert = BERT_model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
    return torch.nn.functional.softmax(ouptput_bert[0], dim=1)[:,0].detach().numpy()
