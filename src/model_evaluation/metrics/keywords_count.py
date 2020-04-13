import numpy as np
import pandas as pd
from gensim.summarization import keywords
from src.model_evaluation.metrics.flexible_metrics import Metrics


class KwCount(Metrics):
    """
    Study the number of identical keywords in P2 and the generated P2.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = keywords

    def __call__(self, predicted_sentences, original_contexts, summarizer):
        """
        :param predicted_sentences: generated P2 by GPT2
        :param original_contexts: original P2
        :param summarizer: name of the summarizer chosen for text generation, among ['T5','BART','PYSUM','KW']
        :return: proportion of number of original P2 keywords existent in generated P2
        """
        df_results = pd.DataFrame(columns=["keyword_proportion"], data=np.zeros((len(predicted_sentences),1)))

        def kw_proportion_one_sentence(pred_P2, kw_list):
            count = len([kw for kw in kw_list if kw is pred_P2.lower().split(' ')])
            return count / len(set(kw_list)) if len(kw_list) != 0 else 'NaN'

        for i, (predicted_sentence, original_context) in enumerate(zip(predicted_sentences, original_contexts)):
            df_results.loc[i, "keyword_proportion"] = kw_proportion_one_sentence(predicted_sentence,
                                                                                 original_context.summaries[summarizer])

        return df_results



class KwIou(Metrics):
    """
    Study the number of identical keywords in P2 and the generated P2.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = keywords

    def __call__(self, predicted_sentences, original_contexts, summarizer):
        """
        :param predicted_sentences: generated P2 by GPT2
        :param original_contexts: original P2
        :return: the iou score between list of keywords in original P2 and generated P2
        """
        df_results = pd.DataFrame(columns=["keyword_proportion"], data=np.zeros((len(predicted_sentences), 1)))

        def kw_iou_one_sentence(predicted_sentence, kw_in_text):
            kw_in_pred = self.model(predicted_sentence, lemmatize=False, pos_filter=('NN', 'JJ', 'VB')).split('\n')
            # kw_in_text = self.model(original_context, lemmatize=False, pos_filter=('NN', 'JJ', 'VB')).split('\n')
            if len(kw_in_pred) == 0 and len(kw_in_text) == 0:
                iou = 'NaN'
            else:
                iou = len(set(kw_in_pred).intersection(kw_in_text)) / len(set(kw_in_pred).union(kw_in_text))
            return iou

        for i, (predicted_sentence, original_context) in enumerate(zip(predicted_sentences, original_contexts)):
            df_results.loc[i, "keyword_proportion"] = kw_iou_one_sentence(predicted_sentence,
                                                                          original_context.summaries[summarizer])

        return df_results
