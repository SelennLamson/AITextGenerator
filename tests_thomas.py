

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
# import random
# from src.json_generation.prepare_data import prepare_data
#
# prepare_data(verbose=1)


# data = json.load(open('data_output/generation_epoch_2_BART.json', 'r'))
# random.shuffle(data)
# data = data[:50]
# json.dump(data, open('data_output/selection_epoch_2_BART.json', 'w'))


#
# ids = ['589', '775', '788']
# ori_files = ['data/smalleval/' + i + '_preproc.json' for i in ids]
# bart_files = ['data/smallevalsum/BART_' + i + '_preproc.json' for i in ids]
# kw_files = ['data/smallevalsum/KW_' + i + '_preproc.json' for i in ids]
#
# for ori_f, bart_f, kw_f in zip(ori_files, bart_files, kw_files):
# 	ori_data = json.load(open(ori_f, 'r', encoding='utf-8'))
# 	bart_data = json.load(open(bart_f, 'r', encoding='utf-8'))
# 	kw_data = json.load(open(kw_f, 'r', encoding='utf-8'))
#
# 	ori_pars = ori_data['paragraphs']
# 	bart_pars = bart_data['paragraphs']
# 	kw_pars = kw_data['paragraphs']
#
# 	for ori_p, bart_p, kw_p in zip(ori_pars, bart_pars, kw_pars):
# 		ori_p['summaries'] = dict()
# 		ori_p['summaries']['BART'] = bart_p['summaries']['BART']
# 		ori_p['summaries']['KW'] = kw_p['summaries']['KW']
#
# 	json.dump(ori_data, open(ori_f, 'w', encoding='utf-8'))
#
#
#
#
#
