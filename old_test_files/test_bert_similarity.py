from src.flexible_models import FlexibleBERTEmbed
from src.model_evaluation.metrics import bert_similarity

if __name__ == '__main__':

    bert_embed = FlexibleBERTEmbed(max_length=200, batch_size=5)

    input_sentences = [
        "Alex hate to play with his friend",
        "Alex like to play with his friend",
        "Alex play football",
        "Alex has seen a football game",
        'Alex like to play football'
    ]

    true_sentences = ["Alex like to play football with his friend"] * len(input_sentences)

    similarities = bert_similarity(input_sentences, true_sentences, bert_embed)
    print(similarities)