from src.model_evaluation import GPT2EvaluationScript
from src.flexible_models import FlexibleGPT2
from src.utils import DEFAULT_DECODING_STRATEGY

from transformers import GPT2LMHeadModel, GPT2Tokenizer

script = GPT2EvaluationScript(file_ids=["36", "370"], batch_size=8)


gpt_2 = FlexibleGPT2(model=GPT2LMHeadModel.from_pretrained('gpt2'),
                     tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
                     decoding_strategy=DEFAULT_DECODING_STRATEGY)

script(generations_path='data/outputs/gpt2_benchmark/generation.json',
       results_path='data/outputs/gpt2_benchmark/metrics.json',
       GPT2_model=gpt_2,
       compute_bert_similarity=True,
       compute_entites_iou=True,
       compute_gpt2_perplexity=True,
       compute_residual_tokens=True,
       verbose=1)

