from src.json_generation.ent_sum_preprocessing import perform_global_ner_on_all
from src.json_generation.paragraph_preprocessing import separate_paragraphs_all_files
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.utils import *


def prepare_data(files: List[str] = None, do_ner=True, do_split=True, verbose=0):
    if files is None:
        if do_ner:
            files = [f[:-len(METADATA_SUFFIX)] for f in os.listdir(METADATA_PATH)]
        else:
            files = [f[:-len(NOVEL_SUFFIX)] for f in os.listdir(NOVEL_PATH)]

    if do_ner:
        if verbose >= 1:
            print("Performing NER...")
        ner_model = FlexibleBERTNER(BERT_NER_LARGE, batch_size=256, max_length=128)
        perform_global_ner_on_all(ner_model, files, verbose=verbose)

    if do_split:
        if verbose >= 1:
            print("Splitting into paragraphs...")
        separate_paragraphs_all_files(True, files, min_length=1400, max_length=1700, skip_begin=0.03, verbose=verbose)

    if verbose >= 1:
        print("Preparation terminated.")
