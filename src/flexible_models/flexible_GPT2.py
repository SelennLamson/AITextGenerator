from .flexible_model import FlexibleModel
from src.utils import GPT2_BLOCK_SIZE
import torch

from src.flexible_models.GPT2_lm_segment_model import GPT2LMSegmentModel


class FlexibleGPT2(FlexibleModel):
    """
    A FlexibleGPT2 model is simply the combination of a huggingface gpt2 transformers model and
    a decoding strategy
    """

    def __init__(self, model, tokenizer, decoding_strategy):
        """
        Initializes a GPT2 model.
        :param model: huggingface gpt2 transformers
        :param tokenizer: huggingface gpt2 tokenizers
        :param decoding_strategy: dict of parameters for huggingface transformers.generate methods
        """
        super().__init__()
        self.model = model
        self.model.eval()
        if torch.cuda.is_available():
            model.cuda()

        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.decoding_strategy = decoding_strategy
        self.max_length = GPT2_BLOCK_SIZE
        self.min_length = 0
        self.set_decoding_strategy(decoding_strategy)

        if isinstance(self.model, GPT2LMSegmentModel):
            self.model.set_special_tokens(self.tokenizer)

    def set_decoding_strategy(self, decoding_strategy):
        self.decoding_strategy = decoding_strategy
        self.max_length = decoding_strategy[
            'max_length'] if 'max_length' in decoding_strategy.keys() else GPT2_BLOCK_SIZE
        self.min_length = decoding_strategy['min_length'] if 'min_length' in decoding_strategy.keys() else 0

    def predict(self, input_ids, nb_samples=1):
        """
        Performs GPT-2 generation on strings of any length.
        :param input_ids: torch.tensor of shape (batch_size, max_length)
        :param nb_samples: nb_sample to generate for each input example
        :return: list of strings of len batch_size * nb_samples
        """

        # If inputs_ids consist of a single example, we create from it a batch of 1 example
        if len(input_ids.shape) == 1:
            input_ids = input_ids.view(1, -1)

        # We use a mask so that GPT2 does not take into account the PAD token during generation time
        mask = (input_ids != self.tokenizer.pad_token_id).long()

        self.decoding_strategy['max_length'] = self.max_length
        self.decoding_strategy['min_length'] = 10

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            mask = mask.cuda()

        outputs_id = self.model.generate(input_ids=input_ids,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id,
                                         attention_mask=mask,
                                         num_return_sequences=nb_samples,
                                         **self.decoding_strategy)

        outputs_id = outputs_id.detach().cpu()

        # only keep the token corresponding to the generation part
        # this is because transformers.generate methods also return the input part
        truncated_outputs_id = outputs_id[:, input_ids.shape[1]:]

        return [self.tokenizer.decode(truncated_outputs_id[i], skip_special_tokens=False).replace('<|endoftext|>', '')
                for i in range(outputs_id.shape[0])]
