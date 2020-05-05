from torch.utils.data import Dataset
from src.torch_loader.vectorize_input import TrainInput
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

		with open(self.path, 'r', encoding='utf-8') as json_file:
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
		with open(self.path, 'r', encoding='utf-8') as json_files:
			data = json.load(json_files)
		P1, P2, P3 = data['paragraphs'][idx:idx + 3]

		training_example = TrainInput(
			P1=P1['text'],
			P2=P2['text'],
			P3=P3['text'],
			summaries=P2['summaries'],
			size=P2['size'],
			genre=data['genre'],
			persons=P2['persons'],
			organisations=P2['organisations'],
			locations=P2['locations'],
			misc=P2['misc']
		)

		return self.transform(training_example)
