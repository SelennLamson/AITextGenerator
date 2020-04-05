

############################################
# THOMAS's TEST FILE, PLEASE DO NOT MODIFY #
############################################

from src.model_use.gpt2_benchmark import GPT2BenchmarkScript

script = GPT2BenchmarkScript(file_ids=["16"], batch_size=8)
# script.generate_texts(generations_path='data/outputs/gpt2_benchmark/generation.json', verbose=1)
script.compute_metrics(generations_path='data/outputs/gpt2_benchmark/generation.json',
					   results_path='data/outputs/gpt2_benchmark/metrics.json',
					   compute_bert_similarity=False,
					   compute_entites_iou=True,
					   compute_gpt2_perplexity=False,
					   compute_residual_tokens=False,
					   verbose=1)
