from src.torch_loader import VectorizeParagraph

class TextGeneration:
    """
    TextGeneration will be used to generate text from input given by webserver
    It combines a FlexibleGPT2 model and a specific vectorizer that will transform
    the data from webserver into the correct inputs id
    """
    def __init__(self, GPT2_model):
        """
        TextGeneration is initialized with a fine-tuned GPT2 model and a decoding_strategy
        :param GPT2_model: FlexibleGPT2
        """
        self.GPT2_model = GPT2_model
        self.vectorizer = VectorizeParagraph(tokenizer=self.GPT2_model.tokenizer, block_size=1020, mode="generation")

    def __call__(self, context_input, nb_samples=1):
        """
        :param context_input: [GenerationInput] instance containing the context for the generation
        :param nb_samples: number of different generated text that will be output
        :return: list[str] : list of generated text
        """
        print(context_input.vectorizer_input_format())
        input_ids = self.vectorizer(context_input.vectorizer_input_format())
        print(self.GPT2_model.tokenizer.decode(input_ids))
        return self.GPT2_model.predict(input_ids, nb_samples)