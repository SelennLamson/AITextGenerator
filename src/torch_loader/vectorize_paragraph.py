import torch

class VectorizeParagraph:
    """
    class VectorizeParagrah :
    from {input: (metadata, P1, P3), target: P2} generate {input: X, output: Y}
    when X and Y are vectorized token tensors obtains using GPT2Tokenizer

    An instance of this class will be callable on {input: (metadata, P1, P3), target: P2}
    Later it will be possible to configure it (for instance for randomly erase some data)
    """
    def __init__(self, tokenizer, block_size=1020, debug_mode=False):
        """
        Initialize GPT2Tokenizer and add specific token
        :param block_size : int, max sequence_token_len for GPT_2 input
            all our tokenized sentenced will be padded to have a block_size len
        """
        self.tokenizer = tokenizer

        # We can add locations / organisations later
        # IMPORTANT : BEFORE FINE-TUNING THE MODEL, WE WILL NEED TO RESIZE THE TOKEN EMBEDDINGS
        # model.resize_token_embeddings(len(tokenizer))

        self.block_size = block_size
        self.debug_mode = debug_mode

    @staticmethod
    def bin_size(size):
        """
        :param size: int <-> size of a paragraph
        :return: special token [S], [M], [L] <-> corresponding binned size
        for now only return [M]
        """
        return ' [M] '

    def __call__(self, sample):
        """
        :param sample: tupple (metadata, P1, P3, P2)
            metadata : dict 'title' -> [str], 'author' -> [str], 'genre' -> list[str]
            P1, P2, P3 : dict
                'size': [int]
                'text' : [str]
                'summaries' : list[str]
                'persons': list[str]

        1/ Concatanate all the information from sample as a global string in the following order
            [P1] P1 [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P2] P2 [EOS]

        2/ Tokenize the result and get the id of each token using GPT2.tokenizer encode method

        3/ Check if the size of the tokenize version is less than block_size.
            If not, in the order :
                - truncate P3 from right
                - truncate P1 from left
                - just use P2 (truncate from right if needed)

        4/ Return the tokenize list as a torch tensor
        """

        metadata, P1, P3, P2 = sample

        input_dict = {'P1': self.tokenizer.encode('[P1] ' + P1['text']),
                      'P3': self.tokenizer.encode('[P3] ' + P3['text']),
                      'Sum': self.tokenizer.encode('[Sum] ' + ""),
                      'metadata': self.tokenizer.encode('[T] ' + " - ".join(metadata['genre']) +
                                                        '[Ent] ' + " - ".join(P2["persons"]) +
                                                        self.bin_size(P2['size'])),
                      'P2': self.tokenizer.encode('[P2] ' + P2['text'] + '[EOS]')}

        def concat(input_d):
            return input_d['P1'] + input_d['P3'] + input_d['Sum'] + input_d['metadata'] + input_d['P2']

        vector_size = sum(map(len, input_dict.values()))

        # If the vector size is not too long, we return it directly
        if vector_size <= self.block_size:
            return torch.tensor(concat(input_dict))

        # Else we truncate the P3
        size_wo_P3 = vector_size - len(input_dict['P3'])
        if size_wo_P3 <= self.block_size:
            input_dict['P3'] = input_dict['P3'][:(self.block_size - size_wo_P3)]
            return torch.tensor(concat(input_dict))

        # If still not sufficient, we remove P1 and truncate P1 from left but still add the [P1] special token
        size_wo_P3_and_P1 = vector_size - len(input_dict['P3']) - len(input_dict['P1'])
        if size_wo_P3_and_P1 <= self.block_size:
            input_dict['P3'] = []
            input_dict['P1'] = self.tokenizer.encode('[P1]') + \
                               input_dict['P1'][self.block_size - size_wo_P3_and_P1+1:]
            return torch.tensor(concat(input_dict))

        # It still to big we simply return truncated P2
        else:
            return torch.tensor(input_dict['P2'][:min(self.block_size, len(input_dict['P2']))])

