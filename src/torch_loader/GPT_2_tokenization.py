from transformers import GPT2Tokenizer

class TransformParagraphs:
    """
    class TransformParagraphes :
    from {input: (metadata, P1, P3), target: P2} generate {input: X, output: Y}
    when X and Y are vectorized token tensors obtains using GPT2Tokenizer

    An instance of this class will be callable on {input: (metadata, P1, P3), target: P2}
    Later it will be possible to configure it (for instance for randomly erase some data)
    """
    def __init__(self, block_size=512):
        """
        Initialize GPT2Tokenizer and add specific token
        :param block_size : int, max sequence_token_len for GPT_2 input
            all our tokenized sentenced will be padded to have a block_size len
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                           'eos_token': '[EOS]',
                                           'additional_special_tokens': ['[P1]', '[P3]', '[S]', ' [M]',
                                                                          '[L]', '[T]', '[Sum]', '[Ent]']})
        # We can add locations / organisations later
        # IMPORTANT : BEFORE FINE-TUNING THE MODEL, WE WILL NEED TO RESIZE THE TOKEN EMBEDDINGS
        # model.resize_token_embeddings(len(tokenizer))

        self.tokenizer.padding_side = "right"  # we will pad the right side
        self.block_size = block_size

    def reduce_string_list(self, string_list):
        """
        :param string_list: list of string ["str1", "str2", ..., "strN"]
        :return: a string "str1 - str2 - .. - strN"
        will be used of entities list
        """
        if len(string_list) == 0:
            return ""
        if len(string_list) == 1:
            return string_list[0]

        return string_list[0] + ", " + self.reduce_string_list(string_list[1:])

    def bin_size(self, size):
        """
        :param size: int <-> size of a paragraph
        :return: special token [S], [M], [L] <-> corresponding binned size
        for now only return [M]
        """
        return ' [M] '

    def __call__(self, sample):
        """
        1/ Concatanate in the order
            [P1] P1 [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P2] P2 [EOS]
        """
        metadata, P1, P3, P2 = sample
        string_input = '[P1] ' + P1['text'] + \
                       '[P3] ' + P3['text'] + \
                       '[T] ' + metadata['theme'] + \
                       '[Ent] ' + self.reduce_string_list(P2["persons"]) + \
                        self.bin_size(P2['size']) + \
                        '[P2] ' + P2['text'] + '[EOS]'
        # '[Sum]' + P2['summaries'] + \  POUR LE MOMENT N'UTILISE PAS LES RÉSUMÉS

        print("input_string", string_input)

        tokenized_input = self.tokenizer.encode(string_input,
                                                max_length=self.block_size,
                                                pad_to_max_length=True)

        return tokenized_input
