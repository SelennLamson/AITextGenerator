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
from src.torch_loader import DatasetFromRepo, VectorizeParagraph, VectorizeMode
from src.flexible_models.flexible_bert_embed import FlexibleBERTEmbed
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.model_evaluation.metrics.bert_similarity import bert_similarity
from src.model_evaluation.metrics.entities_iou import entities_iou
from src.model_evaluation.metrics.gpt2_perplexity import gpt2_perplexity
from src.model_evaluation.metrics.residual_tokens import residual_tokens
from src.model_evaluation.metrics.bert_relationship import bert_relationship


class GPT2EvaluationScript:
    def __init__(self,
                 file_ids: List[str],
                 path_to_data_folder=PREPROC_PATH,
                 batch_size: int = 1,
                 use_context=True,
                 path_to_bert_ner=BERT_NER_LARGE):
        """
        Initializes a GPT-2 Benchmark script that will perform text generation on the paragraphs of given files.
        Call the script using parentheses to launch it.
        :param file_ids: list of book ids from data_folder that will be evaluated
        :param path_to_data_folder : path to the datafolder (by default src.utils.PREPOC_PATH)
        :param batch_size: number of simultaneous text generations + text evalution
                    will be used by all flexible model + metrics
        :param use_context: if True, will create special context sentences for model input :
                    [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
                            else, will juste use P1 without any special tokens
                --> put use_context = False to compute GPT_2 baseline
        :param path_to_bert_ner: path to bert ner model (needed if to use GPT2EvalutionScript to compute entities iou)
        """

        # Filtering file ids on files that really exist in the preproc folder
        self.list_of_fid = [f for f in file_ids if os.path.exists(path_to_data_folder + f + PREPROC_SUFFIX)]

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

    @staticmethod
    def pad_left_side(sequences, padding_value):
        """
        Modification of torch.nn.utils.rnn.pad_sequence so that we pad left side and not right side
        :param sequences : list of tensors
        :param padding_value : tokenizer.pad_token_id
        :return tensor of shape (len(sequences), max_length of sequence in sequences)
                the tensor are padded on the left side using pad_token_id from GPT2 tokenizer
        """
        max_len = max([s.size(0) for s in sequences])
        out_dims = (len(sequences), max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, max_len - length:] = tensor
        return out_tensor

    def custom_collate(self, data_samples, padding_value):
        """
        Will be use to instanciate a torch dataloader from our custom dataset
        :param data_samples: list of sample
            each sample is a tupple ([torch.tensor] input_ids, [str] true_P2)
        :param padding_value : tokenizer.pad_token_id
        :return: a tupple: (concatenation on batch axis using left padding of the input_ids,
                            list of true_P2, P3)
        """
        input_ids = self.pad_left_side([sample[0] for sample in data_samples], padding_value)
        target_P2 = [sample[1] for sample in data_samples]
        P3 = [sample[2] for sample in data_samples]

        return input_ids, target_P2, P3

    def generate_texts(self, generations_path: str, GPT2_model:FlexibleGPT2, verbose: int = 1):
        """Starts the text generation on all paragraphs.
        :param generations_path: The path where text generations should be saved.
        :param GPT2_model: FlexibleGPT2 model that need to be evaluated and will be used to generate text
        :param verbose: 0 for silent execution, 1 for progress.
        """
        vectorizer = VectorizeParagraph(tokenizer=GPT2_model.tokenizer,
                                        mode=VectorizeMode.EVAL,
                                        use_context=self.use_context)

        dataset = DatasetFromRepo(path=PREPROC_PATH, sublist=self.list_of_fid, transform=vectorizer)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                collate_fn=lambda x: self.custom_collate(x, GPT2_model.tokenizer.pad_token_id))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        GPT2_model.model.to(device)

        if verbose:
            print("\rGenerating texts...", end="")

        generations, originals, P3_list = [], [], []
        for input_ids, true_P2, P3 in tqdm(dataloader):
            generations += GPT2_model(input_ids)
            originals += true_P2
            P3_list += P3

        if verbose:
            print("\rSaving generated texts...", end="")

        json_data = [{"generated": gen, "original": ori, "P3": P3}
                     for gen, ori, P3 in zip(generations, originals, P3_list)]

        json.dump(json_data, open(generations_path, 'w', encoding='utf-8'))

        if verbose:
            print("\rGeneration successfull.")

    def compute_metrics(self,
                        generations_path:str,
                        results_path:str,
                        compute_bert_similarity=False,
                        compute_entites_iou=False,
                        compute_gpt2_perplexity=False,
                        compute_residual_tokens=False,
                        compute_bert_relationship=False,
                        verbose: int = 1):
        """Computes the selected metrics on generated texts.
        :param generations_path: The path where text generations can be found.
        :param results_path: The path where results should be saved.
        :param compute_bert_similarity: Should "BERT similarity" metric be computed?
        :param compute_entites_iou: Should "Entities I-o-U" metric be computed?
        :param compute_gpt2_perplexity: Should "GPT-2 perplexity" metric be computed?
        :param compute_residual_tokens: Should "Residual tokens" metric be computed?
        :param compute_bert_relationship: Should "Bert relationship" metric be computed?
        :param verbose: 0 for silent execution, 1 for progress.
        """

        if verbose:
            print("Computing metrics...", end="")

        generations = json.load(open(generations_path, 'r', encoding='utf-8'))
        generated = [g['generated'] for g in generations]
        originals = [g['original'] for g in generations]
        P3 = [g['P3'] for g in generations]

        if os.path.exists(results_path):
            results = json.load(open(results_path, 'r'))
        else:
            results = dict()
            results['per_paragraph'] = [dict() for _ in range(len(originals))]

        per_paragraph = results['per_paragraph']

        def register_stats(_array, _name):
            results[_name] = {'mean': str(np.mean(_array)),
                              'max': str(np.max(_array)),
                              'min': str(np.min(_array)),
                              'median': str(np.median(_array))}

        if compute_bert_similarity:
            bert_embed_model = FlexibleBERTEmbed(2000, self.batch_size)
            bert_similarities = bert_similarity(originals, generated, bert_embed_model, verbose)

            # Freeing space
            del bert_embed_model

            if verbose:
                print("\rRegistering bert simirality results...", end="")
            for i, sim in enumerate(bert_similarities):
                per_paragraph[i]['bert_similarity'] = sim
            register_stats(bert_similarities, 'bert_similarity')

        if compute_entites_iou:
            bert_ner_model = FlexibleBERTNER(self.path_to_bert_ner, batch_size=self.batch_size)

            ent_ious = entities_iou(originals, generated, bert_ner_model)

            ent_ious = np.sum([ent_ious[key] for key in ENTITY_TAGS], axis=0) / len(ENTITY_TAGS)

            # Freeing space
            del bert_ner_model

            if verbose:
                print("\rRegistering entities I-o-U results...", end="")
            for i, ent in enumerate(ent_ious):
                per_paragraph[i]['entities_iou'] = ent
            register_stats(ent_ious, 'entities_iou')

        if compute_gpt2_perplexity:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            flexible_model = FlexibleGPT2(model, tokenizer, DEFAULT_DECODING_STRATEGY)

            gpt2_gen_perplexities = gpt2_perplexity(generated, flexible_model, device)
            gpt2_ori_perplexities = gpt2_perplexity(originals, flexible_model, device)
            gpt2_perplexities = gpt2_gen_perplexities / gpt2_ori_perplexities

            # Freeing space
            del flexible_model
            del tokenizer
            del model

            if verbose:
                print("\rRegistering GPT-2 perplexities results...", end="")
            for i, perplexity in enumerate(gpt2_perplexities):
                per_paragraph[i]['gpt2_perplexity'] = perplexity
            register_stats(gpt2_perplexities, 'gpt2_perplexity')

        if compute_residual_tokens:
            res_toks = residual_tokens(generated)
            if verbose:
                print("\rRegistering Residual Tokens results...", end="")
            for i, res in enumerate(res_toks):
                per_paragraph[i]['residual_tokens'] = float(res)
            register_stats(res_toks, 'residual_tokens')

        json.dump(results, open(results_path, 'w'))

        if compute_bert_relationship:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased').to(device)

            gen_relationships = bert_relationship(generated, P3, model, tokenizer, self.batch_size).astype(float)
            #ori_relationships = bert_relationship(originals, P3, model, tokenizer, self.batch_size)
            #normalized_relationships = gen_relationships / ori_relationships #TODO will do shit if ori == 0

            del tokenizer
            del model

            if verbose:
                print("\rRegistering BERT relationship results...", end="")
            for i, relationship in enumerate(gen_relationships):
                per_paragraph[i]['bert_relationship'] = relationship
            register_stats(gen_relationships, 'bert_relationship')

        print("RESULTS:")
        print(results)
        json.dump(results, open(results_path, 'w'))

        if verbose:
            print("\rMetrics computed successfully.")



