from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.model_evaluation.metrics.flexible_metrics import Metrics
from src.utils import ENTITY_TAGS, ENTITY_CLASSES, BERT_NER_LARGE
import numpy as np
import pandas as pd


class EntitiesCount(Metrics):
    """
    Compute the proportion of entities of true_P2 that is present in pred_P2 for each type of class
    Compute also the total number of absent entities in true_P2 that are present in pred_P2
    """

    def __init__(self, **kwargs):
        """
        Initialized the BERT NER model
        :param path_to_bert_ner: path to the folder containing the BERT NER weights
        :param batch_size: [int] batch size to used for bert
        """
        super().__init__()
        # self.ner_model = FlexibleBERTNER(kwargs['path_to_bert_ner'], batch_size=kwargs['batch_size'])

    def __call__(self, predicted_sentences, original_contexts):
        """
        :param predicted_sentences: list[str] batch of sentences corresponding to the generated P2
        :param original_contexts: list[TrainInput] corresponding to original training examples
        :return: pd.DataFrame containing
            - proportion of correct entities for each class / each pred_P2
        """
        # entities_in_preds = self.ner_model.predict(predicted_sentences, verbose=0)

        df_proportion = pd.DataFrame(columns=["proportion_of_"+class_name for class_name in ENTITY_CLASSES],
                                     data=np.zeros((len(predicted_sentences), len(ENTITY_CLASSES))))

        # def filter_by_class(entity_list, eclass_tag):
        #     return set(entity for entity, value in entity_list.items() if value[0] == eclass_tag)
        #
        # for i, (entities_in_pred, original_context) in enumerate(zip(entities_in_preds, original_contexts)):
        #     for class_tag, class_name in zip(ENTITY_TAGS, ENTITY_CLASSES):
        #         entities_of_class_tag_in_pred = filter_by_class(entities_in_pred, class_tag)
        #         entities_in_class_tag_in_true = set(original_context.to_dict()[class_name])
        #
        #         proportion = -1 if len(entities_in_class_tag_in_true) == 0 \
        #                         else len(entities_in_class_tag_in_true.intersection(entities_of_class_tag_in_pred)) / \
        #                              len(entities_in_class_tag_in_true)
        #
        #         df_proportion.loc[i, "proportion_of_"+class_name] = proportion

        for i, (predicted_sentence, original_context) in enumerate(zip(predicted_sentences, original_contexts)):
            cleaned = predicted_sentence.lower().strip()
            for class_name in ENTITY_CLASSES:
                ents_in_context = set(original_context.to_dict()[class_name])
                prop = -1 if len(ents_in_context) == 0 else len([ent for ent in ents_in_context if ent.lower() in cleaned]) / len(ents_in_context)
                df_proportion.loc[i, "proportion_of_"+class_name] = prop

        return df_proportion
