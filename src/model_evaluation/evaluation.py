from torch.utils.data import DataLoader

class Evaluation:
	def __init__(self, model):
		"""
		:param model: GPT2 model to evaluate
		"""
		self.model = model

	def predict(self, dataset, nb_examples, batch_size=1, **kwargs):
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
