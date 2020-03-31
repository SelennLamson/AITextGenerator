from torch_loader import DatasetFromRepo
import random

"""
Script to test the custom torch dataset module
We test the torch dataset without using vectorization and print some result 
"""

JSON_FILE_PATH = "data/ent_sum/"

if __name__ == '__main__':
	novels_dataset = DatasetFromRepo(path=JSON_FILE_PATH, transform=lambda x: x)

	print('dataset size : ', len(novels_dataset))
	for i in range(10):
		idx = random.randint(0, len(novels_dataset) - 1)
		print('\nExemple n°', i, ':')
		example = novels_dataset[idx]
		metadata, P1, P2, P3 = example
		print('metadata :', metadata)
		print('P1 :', P1)
		print('P2 :', P2)
		print('P3 :', P3)


