from .flexible_model import FlexibleModel

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
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        self.decoding_strategy = decoding_strategy
        self.max_length = decoding_strategy['max_length']

    def predict(self, input_ids, nb_samples=1):
        """
        Performs NER on strings of any length.
        :param input_ids: torch.tensor of shape (batch_size, max_length)
        :param nb_samples: nb_sample to generate for each input example
        :return: list of strings of len batch_size * nb_samples
        """

        # If inputs_ids consist of a single example, we create from it a batch of 1 example
        if len(input_ids.shape) == 1:
            input_ids = input_ids.view(1, -1)

        # We use a mask so that GPT2 does not take into account the PAD token during generation time
        mask = (input_ids != self.tokenizer.pad_token_id).long()

        # TODO : investigate the way transformers.generate take into account max_length
        self.decoding_strategy['max_length'] = self.max_length + input_ids.shape[1]
        outputs_id = self.model.generate(input_ids=input_ids,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         attention_mask=mask,
                                         num_return_sequences=nb_samples,
                                         **self.decoding_strategy)

        # only keep the token corresponding to the generation part
        # this is because transformers.generate methods also return the input part
        truncated_outputs_id = outputs_id[:, input_ids.shape[1]:]

        return [self.tokenizer.decode(truncated_outputs_id[i], skip_special_tokens=True)
                for i in range(outputs_id.shape[0])]
