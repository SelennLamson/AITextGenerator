import torch
from enum import Enum
from src.utils import get_size_from_chars, GPT2_BLOCK_SIZE
from src.torch_loader.vectorize_input import TrainInput, GenerationInput

class VectorizeMode(Enum):
    TRAIN = 0
    EVAL = 1
    GENERATE = 2


# TODO : change documentation with new special tokens
class VectorizeParagraph:
    """
    class VectorizeParagrah
    An instance of this class will be callable on {input: (metadata, P1, P3), target: P2}
    """
    def __init__(self,
                 tokenizer,
                 block_size=GPT2_BLOCK_SIZE,
                 mode=VectorizeMode.TRAIN,
                 use_context=True,
                 select_summary=lambda x:""):
        """
        Store the parameters of the vectorizer, by default : mode = Train, use_context = True
        :param block_size : int, max sequence_token_len for GPT_2 input
        :param mode: VectorizerMode
        :param use_context: True to use full context with special token
                            False to only use P1 without any special token
        :param select_summary (only for input and eval mode) function to select a summary from a dict of summaries
                    input : [dict] {'T5':str, 'BART':str, ..., 'KW':str}  --> output : [str]
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode
        self.use_context = use_context
        self.select_summary = select_summary

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
        return context_d['P3'] + context_d['Sum'] + context_d['T'] + context_d['Per'] + context_d['Org'] + \
               context_d['Loc'] + context_d['Misc'] + context_d['Size'] + context_d['P1'] + context_d['P2']

    def vectorize(self, context, P2, nb_tokens_for_P2):
        """
        :param context: [dict] representing the context
        :param P2: [str] True P2 or '' in generation_mode
        :param nb_tokens_for_P2: nb of tokens in true P2 or nb of tokens to saved for P2 in generation mode
        :return: vectorize version of the dict
        """

        nb_tokens_for_context = sum(map(len, context.values()))  # Context = everything except P2

        if self.mode == VectorizeMode.TRAIN:
            # In train mode, P2 is added to the context and the sentence to predict is simplify the full context
            context['P2'] += self.tokenizer.encode(P2 + ' ' + '<|endoftext|>') if self.use_context \
                             else self.tokenizer.encode(P2)

        # If the context + input space we must left for P2 is too big
        # We let 2/3 of the remaining space for P1 and 1/3 for P3
        if (nb_tokens_for_context + nb_tokens_for_P2 >= self.block_size) and self.use_context:
            initial_vector_size = len(context['Sum'] + context['T'] + context['Per'] + context['Org'] +
                                      context['Loc'] + context['Misc'] + context['Size']) + nb_tokens_for_P2

            nb_tokens_left_for_P1 = int((self.block_size - initial_vector_size) * 2 / 3 - 1)
            nb_tokens_left_for_P3 = int((self.block_size - initial_vector_size) * 1 / 3 - 1)
            context['P1'] = self.tokenizer.encode('[P1] ') + \
                            context['P1'][len(context['P1']) - nb_tokens_left_for_P1:]  # TRUNCATE P1
            context['P3'] = context['P3'][:nb_tokens_left_for_P3]  # TRUNCATE P3

        # if context is desactived we simply have to truncate P1 from left
        if (nb_tokens_for_context + nb_tokens_for_P2 >= self.block_size) and not self.use_context:
            nb_tokens_left_for_P1 = self.block_size - nb_tokens_for_P2
            context['P1'] = context['P1'][len(context['P1']) - nb_tokens_left_for_P1:]

        token_types = {key: [value[0]] * len(value) if len(value) > 0 else [] for key, value in context.items()}

        tensor_input = torch.tensor(self.concat_context(context))
        tensor_types = torch.tensor(self.concat_context(token_types))
        assert len(tensor_input) == len(tensor_types)

        if self.mode == VectorizeMode.TRAIN:
            labels = torch.tensor([-100] * sum(len(v) for k, v in context.items() if k != 'P2') + context['P2'])

            assert len(labels) == len(tensor_input)
            return tensor_input, tensor_types, labels
        else:
            return tensor_input, tensor_types

    def __call__(self, sample):
        """
        Create an input_dict which formats and encodes (using GPT2 tokenizer) the data

        :param sample: [TrainInput] object in train and evaluation mode
                       [GenerationInput] ojbect in generation mode

        :return:
            for train mode:
                with context : [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2] P2 <|endoftext|>
                without context : P1 P2

            for eval mode:
                with context : [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2], sample
                witouht context : P1, sample

            for generate mode:
                with context: [P3] P3 [Sum] Sum_P2 [T] Theme [Ent] list_of_person [Size] [P1] P1 [P2]
                witout context : P1
        """
        if self.mode == VectorizeMode.TRAIN or self.mode == VectorizeMode.EVAL:
            assert type(sample) == TrainInput, 'In train/eval mode, vectorizer input must be of type TrainInput'

        if self.mode == VectorizeMode.GENERATE:
            assert type(sample) == GenerationInput, 'In generation mode, vectorizer input must be of type GenerationInput'

        context = dict()
        if self.use_context:
            context['P1'] = ' [P1] ' + sample.P1 if sample.P1 != "" else ""
            context['P3'] = ' [P3] ' + sample.P3 if sample.P3 != "" else ""
            context['P2'] = ' [P2] '
            context['T'] = ' [T] ' + " - ".join(sample.genre) if sample.genre != [] else ""
            context['Per'] = ' [Per] ' + " - ".join(sample.persons) if sample.persons != [] else ""
            context['Org'] = ' [Org] ' + " - ".join(sample.organisations) if sample.organisations != [] else ""
            context['Misc'] = ' [Misc] ' + " - ".join(sample.misc) if sample.misc != [] else ""
            context['Loc'] = ' [Loc] ' + " - ".join(sample.locations) if sample.locations != [] else ""
            context['Size'] = self.special_token_for_size(sample.size)

            summary = sample.summary if self.mode == VectorizeMode.GENERATE else self.select_summary(sample.summaries)
            context['Sum'] = '[Sum] ' + summary if summary != "" else ""

        if not self.use_context:
            # if the full context must not used, only P1 will be taken into account without any special tokens
            context = {'P1': sample.P1, 'P2': '', 'P3': '', 'Sum': '', 'T': '',
                       'Per': '', 'Org': '', 'Misc': '', 'Loc': '', 'Size': ''}

        # Encode the context
        for key, value in context.items():
            context[key] = self.tokenizer.encode(value)

        # In generate mode, obviously, the true P2 doest not exist
        P2 = sample.P2 if self.mode != VectorizeMode.GENERATE else ""

        # Compute the number of tokens we saved for P2 in the input block
        if self.mode == VectorizeMode.GENERATE:
            nb_tokens_for_P2 = sample.size.mean_tokens + 50
        else:
            nb_tokens_for_P2 = len(self.tokenizer.encode(P2)) + 2  # +1 for [P2] and +1 for <|endoftext|>

        if self.mode == VectorizeMode.TRAIN:
            input_ids, type_ids, labels = self.vectorize(context, P2, nb_tokens_for_P2)
            return input_ids, type_ids, labels

        if self.mode == VectorizeMode.GENERATE:
            input_ids, type_ids = self.vectorize(context, P2, nb_tokens_for_P2)
            return input_ids, type_ids

        if self.mode == VectorizeMode.EVAL:
            input_ids, type_ids = self.vectorize(context, P2, nb_tokens_for_P2)
            return input_ids, type_ids, sample

