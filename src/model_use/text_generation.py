from src.torch_loader import VectorizeParagraph

class TextGeneration:
    """
    TextGeneration will be used to generate text from input given by webserver
    """
    def __init__(self, GPT2_model, GPT2_tokenizer, decoding_strategy):
        """
        TextGeneration is initialized with a fine-tuned GPT2 model and a decoding_strategy
        :param GPT2_model: huggingface GPT2 transformers model
        :param GPT2_tokenizer: huggingface GPT2 tokeziner
        :param decoding_strategy: dict of parameters for huggingface transformers.generate methods
        """
        self.GPT2_model = GPT2_model
        self.GPT2_tokenizer = GPT2_tokenizer
        self.decoding_strategy = decoding_strategy
        self.max_length = decoding_strategy['max_length']
        self.vectorizer = VectorizeParagraph(tokenizer=self.GPT2_tokenizer, block_size=1020, train_mode=False)

    def __call__(self, context_input, nb_samples=1):
        """
        :param context_input: [GenerationInput] instance containing the context for the generation
        :param nb_samples: number of different generated text that will be output
        :return: list[str] : list of generated text
        """

        input_ids, _ = self.vectorizer(context_input.vectorizer_input_format())
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
