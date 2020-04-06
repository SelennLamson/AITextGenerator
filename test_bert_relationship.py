from transformers import BertTokenizer, BertForNextSentencePrediction
from src.model_evaluation.metrics import bert_relationship

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    sentence_1 = "How old are you ?"
    sentence_2_v1 = "The Eiffel Tower is in Paris"
    sentence_2_v2 = "I am 193 years old"

    output = bert_relationship([sentence_1, sentence_1], [sentence_2_v1, sentence_2_v2], model, tokenizer)

    prob_of_succession_v1 = output[0]
    prob_of_succession_v2 = output[1]

    print("Probality of [", sentence_2_v1,"] succeeding to [", sentence_1, "] = %0.2f"%prob_of_succession_v1)
    print("Probality of [", sentence_2_v2,"] succeeding to [", sentence_1, "] = %0.2f"%prob_of_succession_v2)
