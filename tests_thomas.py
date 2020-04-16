

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
import json
# from src.json_generation.prepare_data import prepare_data

# prepare_data(do_ner=False, do_split=True, verbose=0)


# Parsing paragraphs

files = os.listdir(PREPROC_PATH)

for i, f in enumerate(files):
	print('\r{}/{}'.format(i + 1, len(files)), end="")

	if PREPROC_SUFFIX not in f:
		continue

	data = json.load(open(PREPROC_PATH + f, 'r', encoding='utf-8'))
	original_data = json.load(open(NOVEL_PATH + f[:-len(PREPROC_SUFFIX)] + NOVEL_SUFFIX, 'r', encoding='utf-8'))

	all_p = set(v.strip() for v in original_data['persons'].values() if len(v) > 2).difference({'The', 'the'})
	all_o = set(v.strip() for v in original_data['organisations'].values() if len(v) > 2).difference({'The', 'the'})
	all_l = set(v.strip() for v in original_data['locations'].values() if len(v) > 2).difference({'The', 'the'})
	all_m = set(v.strip() for v in original_data['misc'].values() if len(v) > 2).difference({'The', 'the'})

	paragraphs = data['paragraphs']

	for par in paragraphs:
		search_content = par['text'].replace(',', ' ').replace('"', '').replace("'", '').replace(';', '').replace('_', '').replace('”', '').replace('“', '')
		par['persons'] = [p for p in all_p if p in search_content]
		par['organisations'] = [o for o in all_o if o in search_content]
		par['locations'] = [l for l in all_l if l in search_content]
		par['misc'] = [m for m in all_m if m in search_content]

	json.dump(data, open(PREPROC_PATH + f, 'w', encoding='utf-8'))


