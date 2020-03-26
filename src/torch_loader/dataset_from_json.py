from torch.utils.data import Dataset
import json

class DatasetFromJson(Dataset):
    """
    DatasetFromFile :
    generate vectorize samples (P1, P3, context ; P2) from a single JSON file

    main idea :
        1/ will generate (nb_paragraphes - 2) tupple (input, label)
        2/ will apply a GPT_2 tokenization for each input, label
    """
    def __init__(self, path, transform):
        """
        :param path: JSON file
        :param transform: transform: function to transform paragraph into a vectorized form (GPT_2 tokenization
        """
        self.path = path
        self.transform = transform

        with open(self.path) as json_file:
            data = json.load(json_file)
        self.length = len(data['paragraphs']) - 2

    def __len__(self):
        """
        :return: number of paragraphes in the novel - 2
        """
        return self.length

    def __getitem__(self, idx):
        """
        :return: (vectorized P1, P3 + control code ; vectorized P2 when P2 is the paragraph n°idx+2 of the novel
        """
        with open(self.path) as json_files:
            data = json.load(json_files)
        P1, P2, P3 = data['paragraphs'][idx:idx+3]
        metadata = {k: data[k] for k in ('title', 'author', 'theme')}
        sample = (metadata, P1, P3, P2)

        return self.transform(sample)
