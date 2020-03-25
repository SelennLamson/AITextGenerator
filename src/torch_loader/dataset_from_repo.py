from torch.utils.data import Dataset, ConcatDataset
from .dataset_from_json import DatasetFromJson
import os
import json

class DatasetFromRepo(Dataset):
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
        datasets = [DatasetFromJson(path + json_files, transform) for json_files in os.listdir(path)
                                                                      if json_files[-4:] == "json"]
        self.dataset = ConcatDataset(datasets)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

