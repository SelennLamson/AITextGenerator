

############################################
# THOMAS's TEST FILE, PLEASE DO NOT MODIFY #
############################################

# from src.model_use.gpt2_benchmark import GPT2BenchmarkScript

# script = GPT2BenchmarkScript(file_ids=["16"], batch_size=8)
# script(generations_path='data/outputs/gpt2_benchmark/generation.json',
# 	   results_path='data/outputs/gpt2_benchmark/metrics.json',
# 	   compute_bert_similarity=True,
# 	   compute_entites_iou=True,
# 	   compute_gpt2_perplexity=True,
# 	   compute_residual_tokens=True,
# 	   verbose=1)

from src.utils import *
import os
from src.json_generation.prepare_data import prepare_data

prepare_data(do_ner=False, do_split=True, verbose=0)


