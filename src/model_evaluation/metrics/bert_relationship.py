import torch

def bert_relationship(seq_1, seq_2, BERT_model, BERT_tokenizer):
    """
    :param seq_1: [str] first sequence
    :param seq_2: [str] second sequence
    :param BERT_model: transformers.BertForNextSentencePrediction
    :param BERT_tokenizer: transformers.BertTokenizer
    :return: probability that seq_2 is the continuation of seq_1
    """
    encoded_dict = BERT_tokenizer.encode_plus(seq_1, seq_2)
    input_ids = torch.tensor(encoded_dict['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(encoded_dict['token_type_ids'], dtype=torch.long).unsqueeze(0)

    BERT_model.eval()

    ouptput_bert = BERT_model(input_ids=input_ids, token_type_ids=token_type_ids)

    return torch.nn.functional.softmax(ouptput_bert[0], dim=1)[0][0].item()
