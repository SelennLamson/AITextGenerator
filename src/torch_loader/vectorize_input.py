from src.utils import MEDIUM


class TrainInput:
    """
    Use this class to vectorize a triplet of paragraphs in training or evaluation context
    """

    def __init__(self, P1, P2, P3, summaries, genre, size, persons, organisations, locations, misc):
        """
        :param P1: [str]
        :param P3: [str]
        :param P2: [str]
        :param summaries: dict {'T5': str, 'BART':str , ... 'kw':str}
        :param genre: list[str] the genres of the novel
        :param size: int size of P2
        :param persons: list[str] list of entites that the author want GPT2 to use for text generation
        :param persons: list[str] list of person names that the author want GPT2 to use for text generation (optional)
        :param locations: list[str] list of localisations that the author want GPT2 to use for text generation (optional)
        :param organisations : list[str] list of organisation that the author want GPT2 to use for text generation (optional)
        :param misc : list[str] list of misc that the author want GPT2 to use for text generation (optional)
        """
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.summaries = summaries
        self.genre = genre
        self.size = size
        self.persons = persons
        self.organisations = organisations
        self.locations = locations
        self.misc = misc

    def to_dict(self):
        return {
            'P1': self.P1,
            'P2': self.P2,
            'P3': self.P3,
            'summaries': self.summaries,
            'genre': self.genre,
            'size': self.size,
            'persons': self.persons,
            'organisations': self.organisations,
            'locations': self.locations,
            'misc': self.misc
        }

    @classmethod
    def from_dict(cls, dict):
        return cls(
            P1=dict['P1'],
            P2=dict['P2'],
            P3=dict['P3'],
            summaries=dict['summaries'],
            genre=dict['genre'],
            size=dict['size'],
            persons=dict['persons'],
            organisations=dict['organisations'],
            locations=dict['locations'],
            misc=dict['misc']
        )


class GenerationInput:
    """
    Use this class to vectorize a context input in Generation context
    """

    def __init__(self, P1=None, P3=None, summary=None, genre=None, size=MEDIUM,
                 persons=None, locations=None, organisations=None, misc=None):
        """
        :param P1: [str] previous text before position where the generation must begin (optional)
        :param P3: [str] next text that will follow the generated text (optional)
        :param summary: some general idea to describe what the author want GPT2 to generate (optional)
        :param genre: list[str] the genres of the novel (optional)
        :param size: SMALL, MEDIUM, LARGE  (optional)
        :param persons: list[str] list of person names that the author want GPT2 to use for text generation (optional)
        :param locations: list[str] list of locations that the author want GPT2 to use for text generation (optional)
        :param organisations : list[str] list of organisation that the author want GPT2 to use for text generation (optional)
        :param misc : list[str] list of misc that the author want GPT2 to use for text generation (optional)
        """
        self.P1 = P1 if P1 else ""
        self.P3 = P3 if P3 else ""
        self.persons = persons if persons else []
        self.locations = locations if locations else []
        self.organisations = organisations if organisations else []
        self.misc = misc if misc else []
        self.size = size
        self.summary = summary if summary else ""
        self.genre = genre if genre else []
