from src.torch_loader import VectorizeParagraph, VectorizeMode
from src.utils import GPT2_BLOCK_SIZE
class TextGeneration:
    """
    TextGeneration will be used to generate text from input given by webserver
    It combines a FlexibleGPT2 model and a specific vectorizer that will transform
    the data from webserver into the correct inputs id
    """
    def __init__(self, GPT2_model, use_context):
        """
        TextGeneration is initialized with a fine-tuned GPT2 model and a decoding_strategy
        :param GPT2_model: FlexibleGPT2
        :param use_context: [boolean] True to use full context with special tokens
                                      False to only use P1 without any special tokens
        """
        self.GPT2_model = GPT2_model
        self.vectorizer = VectorizeParagraph(tokenizer=self.GPT2_model.tokenizer,
                                             block_size=GPT2_BLOCK_SIZE,
                                             mode=VectorizeMode.GENERATE,
                                             use_context=use_context)

    def __call__(self, context_input, nb_samples=1, verbose=1):
        """
        :param context_input: [GenerationInput] instance containing the context for the generation
        :param nb_samples: number of different generated text that will be output
        :param verbose: 1 for messages, 0 for silent execution
        :return: list[str] : list of generated text
        """
        input_ids = self.vectorizer(context_input)

        if verbose >= 1:
            print(self.GPT2_model.tokenizer.decode(input_ids))

        return self.GPT2_model.predict(input_ids, nb_samples)
