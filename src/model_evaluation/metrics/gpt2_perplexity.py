from src.flexible_models import FlexibleGPT2
import math
import torch

def gpt2_perplexity(sentences, gpt2_model: FlexibleGPT2):
    """
    Score the sentences by GPT2 model :
     -> perplexity of each sentence for GPT2 internal probability distribution
    :param sentences: list[str] batch of sentences
    :param gpt2_model: [FlexibleGPT2] pre-train GPT2 model encapsulate in a FlexibleGPT2 object
    :return: list[float] list of perplexity score
    """
    gpt2_model.model.eval()
    perplexity = []

    with torch.no_grad():
        for sentence in sentences:
            input_ids = gpt2_model.tokenizer.encode(sentence, return_tensors='pt')
            output = gpt2_model.model.forward(input_ids, labels=input_ids)
            cross_entropy_loss = output[0]
            perplexity.append(math.exp(cross_entropy_loss.item()))

    return perplexity

def normalized_gpt2_perplexity(pred_sentences, true_sentences, gpt2_model: FlexibleGPT2):
    return [gpt2_perplexity([pred_sentence], gpt2_model)[0] / gpt2_perplexity([true_sentence], gpt2_model)[0]
            for (pred_sentence, true_sentence) in zip(pred_sentences, true_sentences)]
