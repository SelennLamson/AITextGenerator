from torch.utils.data import Dataset, ConcatDataset
import os
import json

class DatasetFromJsonRepo(Dataset):
    """
    DatasetFromRepo :
    generate vectorize samples (P1, P3, context ; P2) from a repo of JSON files

    main idea :
        1/ we generate a dataset for each JSON file
        2/ we concacenate all this dataset in a big one using ConcatDataset
            (all this operation will be proceed on the fly)
    """
    def __init__(self, path, transform):
        """
        :param path: repository of all JSON files
        :param transform: function to transform paragraph into a vectorized form
        """
        # We only use json file so before scrapping we check if the extention is json
        datasets = [DatasetFromJsonFile(path + json_files, transform) for json_files in os.listdir(path)
                                                                      if json_files[-4:] == "json"]
        self.dataset = ConcatDataset(datasets)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class DatasetFromJsonFile(Dataset):
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
        self.length = len(data['novel']['paragraphs']) - 2

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
        P1, P2, P3 = data['novel']['paragraphs'][idx:idx+3]
        metadata = {k: data['novel'][k] for k in ('title', 'author', 'theme')}
        sample = (metadata, P1, P3, P2)

        return self.transform(sample)
