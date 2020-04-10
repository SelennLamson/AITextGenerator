from src.utils import MEDIUM

class TrainInput:
    """
    Use this class to vectorize a triplet of paragraphs in training or evaluation context
    """
    def __init__(self, P1, P2, P3, summaries, genre, size, entities):
        """
        :param P1: [str]
        :param P3: [str]
        :param P2: [str]
        :param summaries: dict {'T5': str, 'BART':str , ... 'kw':str}
        :param genre: list[str] the genres of the novel
        :param size: int size of P2
        :param entities: list[str] list of entites that the author want GPT2 to use for text generation
        """
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.summaries = summaries
        self.genre = genre
        self.size = size
        self.entities = entities

class GenerationInput:
    """
    Use this class to vectorize a context input in Generation context
    """
    def __init__(self, P1=None, P3=None, summary=None, genre=None, size=MEDIUM, entities=None):
        """
        :param P1: [str] previous text before position where the generation must begin (optional)
        :param P3: [str] next text that will follow the generated text (optional)
        :param summary: some general idea to describe what the author want GPT2 to generate (optional)
        :param genre: list[str] the genres of the novel (optional)
        :param size: SMALL, MEDIUM, LARGE  (optional)
        :param entities: list[str] list of entites that the author want GPT2 to use for text generation (optional)
        """
        self.P1 = P1 if P1 else ""
        self.P3 = P3 if P3 else ""
        self.entities = entities if entities else []
        self.size = size
        self.summary = summary if summary else ""
        self.genre = genre if genre else []
