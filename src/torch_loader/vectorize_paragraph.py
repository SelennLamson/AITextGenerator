import torch
from src.utils import get_size_from_chars, SMALL, MEDIUM, LARGE

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
        :param mode: one value in ['train', 'eval', 'eval_wo_context', 'generation']
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode

    @staticmethod
    def bin_size(size):
        """
        :param size: int <-> size of a paragraph (in number of chars) or SMALL, MEDIUM, LARGE
        :return: special token [S], [M], [L] <-> corresponding binned size
        for now only return [M]
        """
        if type(size) == int:
            size = get_size_from_chars(size)
        if size == SMALL:
            return ' [S] '
        if size == MEDIUM:
            return ' [M] '
        if size == LARGE:
            return ' [L] '

    @staticmethod
    def nb_tokens_to_save_for_P2(size):
        """
        :param size: SMALL, MEDIUM OR LARGE
        :return: int number token that will be save for P2 in input block
        """
        return size.mean_tokens + 50

    @staticmethod
    def concat(input_d):
        return input_d['P3'] + input_d['Sum'] + input_d['T'] + input_d['Ent'] + \
               input_d['Size'] + input_d['P1'] + input_d['P2']

    def train_mode(self, input_dict, P2):
        """
        Concatanate all the information from sample as a global string in the following order
           [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2] P2 [EOS]
        If needed will truncate P3 then P1 so that the tokenize version of the string is smaller that self.block_size

        :param input_dict [dict] representing the context (see VectorizeParagraph.__call__ for further details)
        :param  P2 [str]
        :return: [torch.tensor] tokenize version of the string
                    [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2] P2 [EOS]
        """
        input_dict['P2'] += self.tokenizer.encode(' ' + P2 + '[EOS]')

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

    def eval_mode(self, input_dict, P2, P3):
        """
        Concatanate all the information from sample as a global string in the following order
            [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P1] P1 [P2]
        If needed will truncate P3 then P1 so that the tokenize version of the string is smaller that self.block_size
        :param [dict] input_dict representing the context (see VectorizeParagraph.__call__ for further details)
        :param [str] P2
        :param [str] P3
        :return: tupple :
                    [torch.tensor] tokenize version of the following string ->
                        [P3] P3 [Sum] Sum_P2 [T] Theme [ENT] list_of_person [Size] [P1] P1 [P2]
                    [str] P2
                    [str] P3
        """
        vector_size = sum(map(len, input_dict.values())) + len(self.tokenizer.encode(P2))
        if vector_size <= self.block_size:
            return torch.tensor(self.concat(input_dict)), P2, P3

        # Else we truncate as in train_mode (first P3, then P1)
        size_wo_P3 = vector_size - len(input_dict['P3'])
        if size_wo_P3 <= self.block_size:
            input_dict['P3'] = input_dict['P3'][:(self.block_size - size_wo_P3)]
            return torch.tensor(self.concat(input_dict)), P2, P3

        # If still not sufficient, we remove P1 and truncate P1 from left but still add the [P1] special token
        size_wo_P3_and_P1 = vector_size - len(input_dict['P3']) - len(input_dict['P1'])
        if size_wo_P3_and_P1 <= self.block_size:
            input_dict['P3'] = []
            input_dict['P1'] = self.tokenizer.encode('[P1]') + \
                               input_dict['P1'][self.block_size - size_wo_P3_and_P1 + 1:]
            return torch.tensor(self.concat(input_dict)), P2, P3

        # By default we return nothing
        return torch.tensor(0), "", P3

    def eval_mode_wo_context(self, P1, P2, P3):
        """
        :param P1 [str]
        :param P2 [str]
        :param P3 [str]
        :return: tokenize version of P1, P2 as a string, P3 as string
        if tokenize of P1 + P2 is to big, we truncate P1 at left side
        """
        P1_encoded = self.tokenizer.encode(P1)
        P2_encoded = self.tokenizer.encode(P2)

        if len(P1_encoded) + len(P2_encoded) <= self.block_size:
            return torch.tensor(P1_encoded), P2, P3
        else:
            remaining_space = self.block_size - len(P2_encoded)
            return torch.tensor(P1_encoded[len(P1_encoded) - remaining_space:]), P2, P3

    def generation_mode(self, input_dict, nb_tokens_saved_for_P2=(MEDIUM.mean_tokens+50)):
        """
        Concatanate all the information from sample as a global string in the following order
            [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
        :param input_dict
        :param nb_tokens_saved_for_P2 :  to know how much we let space for P2 generation
        :return:[torch.tensor] tokenize version of the following string
                    [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
        """
        vector_size = sum(map(len, input_dict.values())) + nb_tokens_saved_for_P2
        if vector_size <= self.block_size:
            return torch.tensor(self.concat(input_dict))

        # Else, we let 2/3 of the space for P1 and 1/3 for P3
        initial_vector_size = len(input_dict['Sum'] + input_dict['metadata']) + nb_tokens_saved_for_P2
        nb_tokens_left_for_P1 = int((self.block_size - initial_vector_size) * 2/3 - 1)
        nb_tokens_left_for_P3 = int((self.block_size - initial_vector_size) * 1/3 - 1)
        input_dict['P1'] = self.tokenizer.encode('[P1] ') + \
                           input_dict['P1'][nb_tokens_left_for_P1:]
        input_dict['P3'] = input_dict['P3'][:nb_tokens_left_for_P3]
        return torch.tensor(self.concat(input_dict))

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
        input_dict = dict()
        input_dict['P1'] = ' [P1] ' + P1['text'] if P1['text'] != "" else ""
        input_dict['P3'] = ' [P3] ' + P3['text'] if P3['text'] != "" else ""
        input_dict['P2'] = ' [P2] '
        input_dict['Sum'] = ' [Sum] ' + ". ".join(P2["summaries"]) if P2["summaries"] != [] else ""
        input_dict['T'] = ' [T] ' + " - ".join(metadata['genre']) if metadata['genre'] != [] else ""
        input_dict['Ent'] = ' [Ent] ' + " - ".join(P2['persons']) if P2['persons'] != [] else ""
        input_dict['Size'] = self.bin_size(P2['size'])


        for key, value in input_dict.items():
            input_dict[key] = self.tokenizer.encode(value)

        if self.mode == "train":
            return self.train_mode(input_dict, P2['text'])
        elif self.mode == "eval":
            return self.eval_mode(input_dict, P2['text'], P3['text'])
        elif self.mode == "eval_wo_context":
            return self.eval_mode_wo_context(P1['text'], P2['text'], P3['text'])
        elif self.mode == "generation":
            return self.generation_mode(input_dict, P2['text'].mean_tokens + 50)
