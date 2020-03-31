from src.torch_loader import VectorizeParagraph

class TextGeneration:
    """
    TextGeneration will be used to generate text from input given by webserver
    """
    def __init__(self, GPT2_model, decoding_strategy):
        """
        TextGeneration is initialized with a fine-tuned GPT2 model and a decoding_strategy
        :param GPT2_model: FlexibleGPT2
        :param decoding_strategy: dict of parameters for huggingface transformers.generate methods
        """
        self.GPT2_model = GPT2_model
        self.decoding_strategy = decoding_strategy
        self.max_length = decoding_strategy['max_length']
        self.vectorizer = VectorizeParagraph(tokenizer=self.GPT2_model.tokenizer, block_size=1020, train_mode=False)

    def __call__(self, context_input, nb_samples=1):
        """
        :param context_input: [GenerationInput] instance containing the context for the generation
        :param nb_samples: number of different generated text that will be output
        :return: list[str] : list of generated text
        """

        input_ids = self.vectorizer(context_input.vectorizer_input_format(), mode="generation")
        return self.GPT2_model.predict(input_ids, nb_samples)