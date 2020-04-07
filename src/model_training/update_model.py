"""
Function to define special tokens (ie : CTRL code) and update the model and tokenizer
"""

from src.utils import *

def add_special_tokens(model=None, tokenizer=None):
    """
    update in place the model and tokenizer to take into account special tokens
    :param model: GPT2 model from huggingface
    :param tokenizer: GPT2 tokenizer from huggingface
    """

    size_tokens = [s.token for s in SIZES]

    if tokenizer:
        tokenizer.add_special_tokens({'bos_token': '[P2]',
                                      'additional_special_tokens': ['[P1]',
                                                                    '[P3]',
                                                                    '[T]',
                                                                    '[Sum]',
                                                                    '[Ent]'] + size_tokens})
    if model:
        model.resize_token_embeddings(len(tokenizer))
