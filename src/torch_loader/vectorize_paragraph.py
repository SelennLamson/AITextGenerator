import torch
from enum import Enum
from src.utils import get_size_from_chars, SMALL, MEDIUM, LARGE, GPT2_BLOCK_SIZE

class VectorizeMode(Enum):
    TRAIN = 0
    EVAL = 1
    GENERATE = 2

class VectorizeParagraph:
    """
    class VectorizeParagrah
    An instance of this class will be callable on {input: (metadata, P1, P3), target: P2}
    """
    def __init__(self, tokenizer, block_size=GPT2_BLOCK_SIZE, mode=VectorizeMode.TRAIN, use_context=True):
        """
        Store the parameters of the vectorizer, by default : mode = Train, use_context = True
        :param block_size : int, max sequence_token_len for GPT_2 input
        :param mode: VectorizerMode
        :param use_context: True to use full context with special token
                            False to only use P1 without any special token
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode
        self.use_context = use_context

    @staticmethod
    def special_token_for_size(size):
        """
        :param size: int <-> size of a paragraph (in number of chars) or SMALL, MEDIUM, LARGE
        :return: special token [S], [M], [L] corresponding to the size
        """
        if type(size) == int:
            size = get_size_from_chars(size)
        return ' ' + size.token + ' '

    @staticmethod
    def concat_context(context_d):
        """
        Concatanate the context_dict in the order
            [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
        :param context_d: [dict] containing the context
        :return: concatanate of context_d values in the right order
        """
        return context_d['P3'] + context_d['Sum'] + context_d['T'] + context_d['Ent'] + \
               context_d['Size'] + context_d['P1'] + context_d['P2']

    def vectorize(self, context, P2, nb_tokens_for_P2):
        """
        :param context: [dict] representing the context
        :param P2: [str] True P2 or '' in generation_mode
        :param nb_tokens_for_P2: nb of tokens in true P2 or nb of tokens to saved for P2 in generation mode
        :return: vectorize version of the dict
        """
        nb_tokens_for_context = sum(map(len, context.values()))  # Context = everything except P2

        if self.mode == VectorizeMode.GENERATE:
            # If the context + input space we must left for P2 is too big
            # We let 2/3 of the remaining space for P1 and 1/3 for P3
            if nb_tokens_for_context + nb_tokens_for_P2 >= self.block_size:
                initial_vector_size = len(context['Sum'] + context['metadata']) + nb_tokens_for_P2
                nb_tokens_left_for_P1 = int((self.block_size - initial_vector_size) * 2 / 3 - 1)
                nb_tokens_left_for_P3 = int((self.block_size - initial_vector_size) * 1 / 3 - 1)
                context['P1'] = self.tokenizer.encode('[P1] ') + context['P1'][nb_tokens_left_for_P1:]  # TRUNCATE P1
                context['P3'] = context['P3'][:nb_tokens_left_for_P3]  # TRUNCATE P3

        if self.mode == VectorizeMode.TRAIN:
            # In train mode, P2 is added to the context and the sentence to predict is simplify the full context
            context['P2'] += self.tokenizer.encode(' ' + P2 + ' [EOS]')

        if self.mode == VectorizeMode.EVAL or self.mode == VectorizeMode.TRAIN:
            # If the context +  P2 is too big, we first truncate P3 then P1
            size_wo_P3 = nb_tokens_for_context + nb_tokens_for_P2 - len(context['P3'])
            if size_wo_P3 > self.block_size: # TODO CHANGE THAT IT IS FALSE !!!! !
                context['P3'] = context['P3'][:(self.block_size - size_wo_P3)]  # TRUNCATE P3

            size_wo_P3_and_P1 = nb_tokens_for_context + nb_tokens_for_P2 - len(context['P3']) - len(context['P1'])
            # If still not sufficient, we remove P1 and truncate P1 from left but still add the [P1] special token
            if size_wo_P3_and_P1 > self.block_size:
                context['P3'] = []
                context['P1'] = self.tokenizer.encode('[P1]') + context['P1'][self.block_size - size_wo_P3_and_P1 + 1:]

        return torch.tensor(self.concat_context(context))

    def __call__(self, sample):
        """
        Create an input_dict which formats and encodes (using GPT2 tokenizer) the data

        :param sample: tupple (metadata, P1, P3, P2)
            metadata : dict 'title' -> [str], 'author' -> [str], 'genre' -> list[str]
            P1, P2, P3 : dict
                'size': [int]
                'text' : [str]
                'summaries' : list[str]
                'persons': list[str]

        :return:
            for train mode:
                with context : [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2] P2 [EOS]
                without context : P1 P2

            for eval mode:
                with context : [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2], P2_str, P3_str
                witouht context : P1, P2_str, P3_str

            for generate mode:
                with context: [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
                witout context : P1
        """
        metadata, P1, P3, P2 = sample

        if self.use_context:
            context = dict()
            context['P1'] = ' [P1] ' + P1['text'] if P1['text'] != "" else ""
            context['P3'] = ' [P3] ' + P3['text'] if P3['text'] != "" else ""
            context['P2'] = ' [P2] '
            context['Sum'] = ' [Sum] ' + ". ".join(P2["summaries"]) if P2["summaries"] != [] else ""  # TODO SELECT ONE RANDOMLY
            context['T'] = ' [T] ' + " - ".join(metadata['genre']) if metadata['genre'] != [] else ""
            context['Ent'] = ' [Ent] ' + " - ".join(P2['persons']) if P2['persons'] != [] else ""
            context['Size'] = self.special_token_for_size(P2['size'])
        else:
            # if the full context must not used, only P1 will be taken into account without any special tokens
            context = {'P1': P1['text'], 'P2': '', 'P3': '', 'Sum': '', 'T': '', 'Ent': '', 'Size': ''}

        # Encode the context
        for key, value in context.items():
            context[key] = self.tokenizer.encode(value)

        # In generate mode, obviously, the true P2 doest not exist
        P2_text = P2['text'] if self.mode != VectorizeMode.GENERATE else ""
        P3_text = P3['text']

        # Compute the number of tokens we saved for P2 in the input block
        if self.mode == VectorizeMode.GENERATE:
            nb_tokens_for_P2 = P2['size'].mean_tokens + 50
        else:
            nb_tokens_for_P2 = len(self.tokenizer.encode(P2_text))

        input_ids = self.vectorize(context, P2_text, nb_tokens_for_P2)

        if self.mode == VectorizeMode.TRAIN or self.mode == VectorizeMode.GENERATE:
            return input_ids
        if self.mode == VectorizeMode.EVAL:
            return input_ids, P2_text, P3_text

