"""
Function to define special tokens (ie : CTRL code) and update the model and tokenizer
"""


def add_special_tokens(model=None, tokenizer=None):
    """
    update in place the model and tokenizer to take into account special tokens
    :param model: GPT2 model from huggingface
    :param tokenizer: GPT2 tokenizer from huggingface
    """

    if tokenizer:
        tokenizer.add_special_tokens(
            {'bos_token': '[P2]',
             'additional_special_tokens': ['[P1]', '[P3]', '[S]', '[M]', '[L]', '[T]', '[Sum]',
                                           '[Loc]', '[Per]', '[Org]', '[Misc]']}
        )
        tokenizer.pad_token = tokenizer.eos_token

    if model:
        model.resize_token_embeddings(len(tokenizer))
