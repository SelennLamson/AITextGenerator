

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


import json
import random

data = json.load(open('data_output/generation_epoch_2_BART.json', 'r'))
random.shuffle(data)
data = data[:50]
json.dump(data, open('data_output/selection_epoch_2_BART.json', 'w'))





