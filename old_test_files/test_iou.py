from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.model_evaluation.metrics import entities_iou
BERT_NER_BASE = 'models/entity_recognition/bert_ner_base/'

if __name__ == "__main__":
    bert_ner = FlexibleBERTNER(BERT_NER_BASE, batch_size=5, max_length=2000)
    input_strings = ["Gael eats potatoes with Alex in Paris", "Thomas eats some fish"]
    output_strings = ["Gael eats potatoes in Paris", "Alex eats some fish"]
    iou = entities_iou(true_paragraphs=input_strings, pred_paragraphs=output_strings,
                       ner_model=bert_ner, class_tags=["PER", "ORG"])

    print(iou)
