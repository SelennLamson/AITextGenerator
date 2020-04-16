import numpy as np
import pandas as pd
import os
import itertools
import argparse
FOLDER_PATH = 'data/results/'


def model_comparison(FOLDER_PATH):
	"""
	:param FOLDER_PATH: path leading to the files (dataframes with metrics results) that we would like to compare.
	Compare the eval metrics obtained for different gpt2 finetuning models (with summary, no summary, no context,...)
	"""
	files = os.listdir(FOLDER_PATH)
	df_list = []
	for i, file in enumerate(files):
		df = pd.read_csv(FOLDER_PATH + file)
		df = df.iloc[:, 1:]  # Remove index column
		df_list.append(df)

	list_models = list(itertools.combinations(range(len(files)), 2))
	for pair in list_models:
		df0 = df_list[pair[0]]
		df1 = df_list[pair[1]]
		cf_df = df1 - df0
		res = cf_df.mean(axis=0)
		print('Difference between', files[pair[0]], 'and', files[pair[1]])
		print(res)


def model_evolution(FOLDER_PATH):
	"""
	:param FOLDER_PATH: path leading to the files (dataframes with metrics results) that we would like to compare.
	Compare the evolution of metrics for a same model but at different epochs
	"""
	# Read files
	for i, file in enumerate(files):
		print('Finetuning evolution of: ', file)
		df = pd.read_csv(FOLDER_PATH + file)
		df = df.iloc[:, 1:]  # Remove index column
		df_list.append(df)

		# Study evolution of metrics within a single dataset - first vs last part
		if len(df) < 5:
			size = len(df) // 2
		else:
			size = len(df) // 5
		# df = df.replace(-1, np.nan)
		# res0 = np.nanmean(df[size:], axis=0)
		res0 = df[size:].mean(axis=0)
		res1 = df[-size:].mean(axis=0)
		results = res1 - res0
		print(results)


# ADD A FILTER (SUFFIXES, PREFIX) to identify files where metrics belong to same model, with same epoch but different models
# ADD ARGUMENTS to specify what epoch to compare or list of models that you want to compare
# Store in dico with key = name model and values = epoch for second function

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--Data", type=str, required=True, help="Path where files are stored")
	args = parser.parse_args()
	model_comparison(args.Data)
	# model_evolution(args.Data)


