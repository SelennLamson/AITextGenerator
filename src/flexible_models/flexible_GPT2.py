from .flexible_model import FlexibleModel

class FlexibleGPT2(FlexibleModel):
    def __init__(self, gpt2_model, gpt2_tokenizer, decoding_strategy):
        """
        Initializes a GPT2 model.
        :param gpt2_model: huggingface gpt2 transformers
        :param gpt2_tokenizer: huggingface gpt2 tokenizers
        :param decoding_strategy: dict of parameters for huggingface transformers.generate methods
        """
        super().__init__()
        self.GPT2_model = gpt2_model
        self.GPT2_tokenizer = gpt2_tokenizer
        self.decoding_strategy = decoding_strategy
        self.max_length = decoding_strategy['max_length']

    def predict(self, input_ids, nb_samples=1):
        """
        Performs NER on strings of any length.
        :param input_ids: torch.tensor of shape (batch_size, max_length)
        :param nb_samples: nb_sample to generate for each input example
        :return: list of strings of len batch_size * nb_samples
        """
        mask = (input_ids != self.GPT2_tokenizer.pad_token_id).long()
        self.decoding_strategy['max_length'] = self.max_length + input_ids.shape[1]
        outputs_id = self.GPT2_model.generate(input_ids=input_ids,
                                              pad_token_id=self.GPT2_tokenizer.eos_token_id,
                                              attention_mask=mask,
                                              num_return_sequences=nb_samples,
                                              **self.decoding_strategy)

        # only keep the token corresponding to the generation part
        # this is because transformers.generate methods also return the input part
        truncated_outputs_id = outputs_id[:, input_ids.shape[1]:]

        return [self.GPT2_tokenizer.decode(truncated_outputs_id[i], skip_special_tokens=True)
                for i in range(outputs_id.shape[0])]
