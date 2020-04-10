from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.model_evaluation.metrics import Metrics
from src.utils import ENTITY_TAGS, BERT_NER_LARGE
import numpy as np

class EntitiesCount(Metrics):
    """
    Compute the proportion of entities of true_P2 that is present in pred_P2 for each type of class
    Compute also the total number of absent entities in true_P2 that are present in pred_P2
    """

    def __init__(self, path_to_bert_ner=BERT_NER_LARGE, batch_size=1):
        """
        Initialized the BERT NER model
        :param path_to_bert_ner: path to the folder containing the BERT NER weights
        :param batch_size: [int] batch size to used for bert
        """
        super().__init__()
        self.ner_model = FlexibleBERTNER(path_to_bert_ner, batch_size=batch_size)

    def __call__(self, predicted_sentences, original_contexts):
        """
        :param predicted_sentences: list[str] batch of sentences
        :param original_contexts: list[TrainInput] correspoing original training example
        :return: pd.DataFrame["relationship"]
        """
        entities_in_pred = self.ner_model.predict(predicted_sentences, verbose=0)

	# TODO CONTINUE HERE


def entities_iou(true_paragraphs, pred_paragraphs, ner_model: FlexibleBERTNER, class_tags=None):
    """Evaluates the intersection-over-union of entities in input and output P2,
            for each example in a batch
            for each class of entity in classes

    :param true_paragraphs: list of true paragraph as a string
    :param pred_paragraphs: list of generated paragraph as a string.
    :param ner_model: FlexibleBERTNER model to detect entities in output P2
    :param class_tags: sublist of  ["PER", "LOC", "ORG, "MISC"], if None compute for all classes

    :return dict {class -> np.array : intersection-over-union score on required and detected entities for each example}
    """
    if class_tags is None:
        class_tags = ENTITY_TAGS

    entities_in_true = ner_model.predict(true_paragraphs, verbose=0)
    entities_in_pred = ner_model.predict(pred_paragraphs, verbose=0)

    def filter_by_class(entity_list, class_tag):
        return set(entity for entity, value in entity_list.items() if value[0] == class_tag)

    def iou(true, pred, single_class):
        true_entities = filter_by_class(true, single_class)
        pred_entities = filter_by_class(pred, single_class)
        union = pred_entities.union(set(true_entities))
        intersection = pred_entities.intersection(set(true_entities))
        return 1 if len(union) == 0 else len(intersection) / len(union)

    def iou_on_batch_for_class(class_tag):
        return np.array(list(map(lambda x:iou(x[0], x[1], class_tag), zip(entities_in_true, entities_in_pred))))

    return {key: iou_on_batch_for_class(key) for key in class_tags}
