import torch

class VectorizeParagraph:
    """
    class VectorizeParagrah :
    from {input: (metadata, P1, P3), target: P2} generate {input: X, output: Y}
    when X and Y are vectorized token tensors obtains using GPT2Tokenizer

    An instance of this class will be callable on {input: (metadata, P1, P3), target: P2}
    Later it will be possible to configure it (for instance for randomly erase some data)
    """
    def __init__(self, tokenizer, block_size=1020, mode='train'):
        """
        Initialize GPT2Tokenizer and add specific token
        :param block_size : int, max sequence_token_len for GPT_2 input
            all our tokenized sentenced will be padded to have a block_size len
        :param mode: one value in ['train', 'eval', 'generation']
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode

    @staticmethod
    def bin_size(size):
        """
        :param size: int <-> size of a paragraph
        :return: special token [S], [M], [L] <-> corresponding binned size
        for now only return [M]
        """
        return ' [M] '

    @staticmethod
    def concat(input_d):
        return input_d['P1'] + input_d['P3'] + input_d['Sum'] + input_d['metadata'] + input_d['P2']

    def train_mode(self, input_dict, P2):
        """
        Concatanate all the information from sample as a global string in the following order
            [P1] P1 [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P2] P2 [EOS]
        If needed will truncate P3 then P1 so that the tokenize version of the string is smaller that self.block_size

        :param input_dict [dict] representing the context (see VectorizeParagraph.__call__ for further details)
        :param  P2 [str]
        :return: [torch.tensor] tokenize version of the string
                    [P1] P1 [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P2] P2 [EOS]
        """
        input_dict['P2'] += self.tokenizer.encode(P2['text'] + '[EOS]')

        vector_size = sum(map(len, input_dict.values()))

        # If the vector size is not too long, we return it directly
        if vector_size <= self.block_size:
            return torch.tensor(self.concat(input_dict))

        size_wo_P3 = vector_size - len(input_dict['P3'])
        # Else we truncate the P3
        if size_wo_P3 <= self.block_size:
            input_dict['P3'] = input_dict['P3'][:(self.block_size - size_wo_P3)]
            return torch.tensor(self.concat(input_dict))

        size_wo_P3_and_P1 = vector_size - len(input_dict['P3']) - len(input_dict['P1'])
        # If still not sufficient, we remove P1 and truncate P1 from left but still add the [P1] special token
        if size_wo_P3_and_P1 <= self.block_size:
            input_dict['P3'] = []
            input_dict['P1'] = self.tokenizer.encode('[P1]') + \
                               input_dict['P1'][self.block_size - size_wo_P3_and_P1 + 1:]
            return torch.tensor(self.concat(input_dict))

        # It still to big we simply return truncated P2
        # This case will normally never happen if the dataset have been correctly constructed
        else:
            input_id = input_dict['P2'][:min(self.block_size, len(input_dict['P2']))]
            if self.train_mode:
                return torch.tensor(input_id)

    def eval_mode(self, input_dict, P2):
        """
        Concatanate all the information from sample as a global string in the following order
            [P1] P1 [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P2]
        If needed will truncate P3 then P1 so that the tokenize version of the string is smaller that self.block_size
        :param [dict] input_dict representing the context (see VectorizeParagraph.__call__ for further details)
        :param [str] P2
        :return: tupple :
                    [torch.tensor] tokenize version of the string
                        P1] P1 [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P2],
                    [str] P2
        """
        # TODO CHECK THE FOLLOWING EQUATION
        vector_size = sum(map(len, input_dict.values())) + len(self.tokenizer.encode(P2))
        if vector_size <= self.block_size:
            return torch.tensor(self.concat(input_dict)), P2

        # Else we truncate as in train_mode (first P3, then P1)
        size_wo_P3 = vector_size - len(input_dict['P3'])
        if size_wo_P3 <= self.block_size:
            input_dict['P3'] = input_dict['P3'][:(self.block_size - size_wo_P3)]
            return torch.tensor(self.concat(input_dict)), P2

        # If still not sufficient, we remove P1 and truncate P1 from left but still add the [P1] special token
        size_wo_P3_and_P1 = vector_size - len(input_dict['P3']) - len(input_dict['P1'])
        if size_wo_P3_and_P1 <= self.block_size:
            input_dict['P3'] = []
            input_dict['P1'] = self.tokenizer.encode('[P1]') + \
                               input_dict['P1'][self.block_size - size_wo_P3_and_P1 + 1:]
            return torch.tensor(self.concat(input_dict)), P2["text"]

        # By default we return nothing
        return torch.tensor(0), ""

    def generation_mode(self, input_dict):
        pass

    def __call__(self, sample):
        """
        Create an input_dict which format and encode (using GPT2 tokenizer) the data
        and pass it to train_model, eval_mode or generation_mode depending of self.mode

        :param sample: tupple (metadata, P1, P3, P2)
            metadata : dict 'title' -> [str], 'author' -> [str], 'genre' -> list[str]
            P1, P2, P3 : dict
                'size': [int]
                'text' : [str]
                'summaries' : list[str]
                'persons': list[str]
        """
        metadata, P1, P3, P2 = sample
        input_dict = {'P1': self.tokenizer.encode('[P1] ' + P1['text']),
                      'P3': self.tokenizer.encode('[P3] ' + P3['text']),
                      'Sum': self.tokenizer.encode('[Sum] ' + ""),
                      'metadata': self.tokenizer.encode('[T] ' + " - ".join(metadata['genre']) +
                                                        '[Ent] ' + " - ".join(P2["persons"]) +
                                                        self.bin_size(P2['size'])),
                      'P2': self.tokenizer.encode('[P2] ')}

        if self.mode == "train":
            return self.train_mode(input_dict, P2['text'])
        elif self.mode == "eval":
            return self.eval_mode(input_dict, P2['text'])
        elif self.mode == "generation":
            return self.generation_mode(input_dict)