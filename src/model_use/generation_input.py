from src.utils import MEDIUM
class GenerationInput:
    """
    GenerationInput will be used as an interface between the webserver backend and our fine-tuned GPT2 model
    """
    def __init__(self, P1=None, P3=None, context=None, genre=None, size=MEDIUM, entities=None):
        """
        :param P1: [str] previous text before position where the generation must begin
        :param P3: [str] next text that will follow the generated text
        :param context: some general idea to describe what the author want GPT2 to generate
        :param genre: list[str] the genres of the novel
        :param size: SMALL, MEDIUM, LARGE
        :param entities: list[str] list of entites that the author want GPT2 to use for text generation
        """
        self.P1 = P1 if P1 else ""
        self.P3 = P3 if P3 else ""
        self.entities = entities if entities else []
        self.size = size
        self.context = context if context else ""
        self.genre = genre if genre else []

    def vectorizer_input_format(self):
        """
        :return: tupple representing self so that can directly be sent to a Vectorizer object
        """
        return ({'genre': self.genre},  # METADATA
                {'text':self.P1},  # P1
                {'text': self.P3},  # P3
                {'size': self.size,
                 'persons': self.entities,
                 'summaries': [self.context]})
