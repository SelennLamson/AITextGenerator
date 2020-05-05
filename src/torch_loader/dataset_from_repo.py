from torch.utils.data import Dataset, ConcatDataset
from .dataset_from_json import DatasetFromJson
from src.utils import PREPROC_SUFFIX
import os


class DatasetFromRepo(Dataset):
	"""

	DatasetFromRepo :
	generate vectorize samples (P1, P3, context ; P2) from a repo of JSON files or a list of JSON files

	main idea :
		1/ we generate a dataset for each JSON file
		2/ we concacenate all this dataset in a big one using ConcatDataset
			(all this operation will be proceed on the fly)

	"""

	def __init__(self, path, sublist=None, transform=lambda x: x):
		"""
		:param path: repository containing json files
		:param sublist: use it to specify a sublist of book_id from the folder in path
		:param transform: function to transform paragraph into a vectorized form
		"""
		# We only use json file so before scrapping we check if the extention is json
		if sublist is None:
			list_of_json_files = [json_file for json_file in os.listdir(path) if json_file[-4:] == "json"]
		else:
			list_of_json_files = [book_name + PREPROC_SUFFIX for book_name in sublist]

		datasets = [DatasetFromJson(path + json_file, transform) for json_file in list_of_json_files]
		self.dataset = ConcatDataset(datasets)

	def __getitem__(self, idx):
		return self.dataset[idx]

	def __len__(self):
		return len(self.dataset)
