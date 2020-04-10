import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer, BertForNextSentencePrediction

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils import *
from src.torch_loader import DatasetFromRepo, VectorizeParagraph, VectorizeMode, TrainInput
from src.flexible_models.flexible_bert_embed import FlexibleBERTEmbed
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.model_evaluation.metrics.bert_similarity import BertSimilarity
from src.model_evaluation.metrics.entities_count import EntitiesCount
from src.model_evaluation.metrics.gpt2_perplexity import GPT2Perplexity
from src.model_evaluation.metrics.bert_relationship import BertRelationship


class GPT2EvaluationScript:
    def __init__(self,
                 file_ids: List[str] = None,
                 path_to_data_folder=PREPROC_PATH,
                 batch_size: int = 1,
                 use_context=True,
                 path_to_bert_ner=BERT_NER_LARGE):
        """
        Initializes a GPT-2 Benchmark script that will perform text generation on the paragraphs of given files.
        Call the script using parentheses to launch it.
        :param file_ids: list of book ids from data_folder that will be evaluated
            (optionnal, if None will compute on every novel in the data_folder)
        :param path_to_data_folder : path to the datafolder (by default src.utils.PREPOC_PATH)
        :param batch_size: number of simultaneous text generations + text evalution
                    will be used by all flexible model + metrics
        :param use_context: if True, will create special context sentences for model input :
                    [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
                            else, will juste use P1 without any special tokens
                --> put use_context = False to compute GPT_2 baseline
        :param path_to_bert_ner: path to bert ner model (needed if to use GPT2EvalutionScript to compute entities iou)
        """

        self.data_folder = path_to_data_folder
        self.list_of_fid = file_ids

        if self.list_of_fid:
            # Filtering file ids on files that really exist in the preproc folder
            self.list_of_fid = [f for f in file_ids if os.path.exists(self.data_folder + f + PREPROC_SUFFIX)]

        self.batch_size = batch_size
        self.use_context = use_context
        self.path_to_bert_ner = path_to_bert_ner

    def __call__(self,
                 generations_path:str,
                 results_path:str,
                 GPT2_model: FlexibleGPT2,
                 compute_bert_similarity=False,
                 compute_entites_iou=False,
                 compute_gpt2_perplexity=False,
                 compute_residual_tokens=False,
                 verbose=1):
        """
        Generates texts at generation_path and computes given metrics on them.
        :param generations_path: The path where text generations can be found.
        :param results_path: The path where results should be saved.
        :param GPT2_model: FlexibneGPT2 model that need to be evaluated and will be used to generate text
        :param compute_bert_similarity: Should "BERT similarity" metric be computed?
        :param compute_entites_iou: Should "Entities I-o-U" metric be computed?
        :param compute_gpt2_perplexity: Should "GPT-2 perplexity" metric be computed?
        :param compute_residual_tokens: Should "Residual tokens" metric be computed?
        :param verbose: 0 for silent execution, 1 for progress.
        """
        self.generate_texts(generations_path, GPT2_model, verbose)
        self.compute_metrics(generations_path, results_path, compute_bert_similarity, compute_entites_iou,
                             compute_gpt2_perplexity, compute_residual_tokens, verbose)

    def generate_texts(self, generations_path: str, GPT2_model:FlexibleGPT2, verbose: int = 1):
        """Starts the text generation on all paragraphs.
        :param generations_path: The path where text generations should be saved.
        :param GPT2_model: FlexibleGPT2 model that need to be evaluated and will be used to generate text
        :param verbose: 0 for silent execution, 1 for progress.
        """
        vectorizer = VectorizeParagraph(tokenizer=GPT2_model.tokenizer,
                                        mode=VectorizeMode.EVAL,
                                        use_context=self.use_context)

        dataset = DatasetFromRepo(path=self.data_folder, sublist=self.list_of_fid, transform=vectorizer)

        def custom_collate(data_samples):
            collate_input_ids = pad_left_side([single_sample[0] for single_sample in data_samples],
                                              padding_value=GPT2_model.tokenizer.pad_token_id)
            original_samples = [single_sample[1] for single_sample in data_samples]
            return collate_input_ids, original_samples

        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=custom_collate)

        if verbose:
            print("\rGenerating texts...", end="")

        generations, originals = [], []
        for input_ids, sample in tqdm(dataloader):
            generations += GPT2_model(input_ids)
            originals += sample

        if verbose:
            print("\rSaving generated texts...", end="")

        json_data = [{"generated": gen, "original": ori.to_dict()} for gen, ori in zip(generations, originals)]
        json.dump(json_data, open(generations_path, 'w', encoding='utf-8'))

        if verbose:
            print("\rGeneration successfull.")

    def compute_metrics(self,
                        generations_path:str,
                        results_path:str,
                        compute_bert_similarity=False,
                        compute_entites_count=False,
                        compute_gpt2_perplexity=False,
                        compute_residual_tokens=False,
                        compute_bert_relationship=False,
                        verbose: int = 1):
        """Computes the selected metrics on generated texts.
        :param generations_path: The path where text generations can be found.
        :param results_path: The path where results should be saved.
        :param compute_bert_similarity: Should "BERT similarity" metric be computed?
        :param compute_entites_count: Should "Entities I-o-U" metric be computed?
        :param compute_gpt2_perplexity: Should "GPT-2 perplexity" metric be computed?
        :param compute_residual_tokens: Should "Residual tokens" metric be computed?
        :param compute_bert_relationship: Should "Bert relationship" metric be computed?
        :param verbose: 0 for silent execution, 1 for progress.
        """

        if verbose:
            print("Computing metrics...", end="")

        generations = json.load(open(generations_path, 'r', encoding='utf-8'))
        generated_sentences = [g['generated'] for g in generations]
        original_contexts = [TrainInput.from_dict(g['original']) for g in generations]

        results = []
        if compute_bert_similarity:
            bert_similarities = BertSimilarity(self.batch_size)(generated_sentences, original_contexts)
            results.append(bert_similarities)

        if compute_entites_count:
            entities_count = EntitiesCount(self.batch_size, self.path_to_bert_ner)(generated_sentences, original_contexts)
            results.append(entities_count)

        if compute_gpt2_perplexity:
            gpt2_perplexity = GPT2Perplexity()(generated_sentences, original_contexts)
            results.append(gpt2_perplexity)

        if compute_bert_relationship:
            bert_relationship = BertRelationship(self.batch_size)(generated_sentences, original_contexts)
            results.append(bert_relationship)

        pd.DataFrame()