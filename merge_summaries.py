import os
import json
from src.utils import *

def merge_summaries():
	"""
	Add summaries for each paragraph of every book.
	We merge the 4 distinct summaries obtained from the summarizers (T5, BART, PYSUM, KW) in our preproc data files
	These summaries are stored as json files (like preproc) in a different folder per summarizer
	The final format in preproc files is a {'summarizer_name':'summary',...}
	"""
	# Loop on all books. Look at original json file + corresponding files but with a summary (x4)
	files = os.listdir(PREPROC_PATH)
	for f in files:
		if PREPROC_SUFFIX in f:
			d_id = f[:-len(PREPROC_SUFFIX)]
			data = json.load(open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'r'))
			data_t5 = json.load(open(FOLDER_NAME_T5 + PREFIX_T5 + d_id + PREPROC_SUFFIX, 'r'))
			data_bart = json.load(open(FOLDER_NAME_BART + PREFIX_BART + d_id + PREPROC_SUFFIX, 'r'))
			data_pysum = json.load(open(FOLDER_NAME_PYSUM + PREFIX_PYSUM + d_id + PREPROC_SUFFIX, 'r'))
			data_kw = json.load(open(FOLDER_NAME_KW + PREFIX_KW + d_id + PREPROC_SUFFIX, 'r'))

			# Add summary from each summariser to the original preproc json file
			for i in range(len(data['paragraphs'])):
				data['paragraphs'][i]['summaries'] = dict()
				data['paragraphs'][i]['summaries'].update(data_t5['paragraphs'][i]['summaries'])
				data['paragraphs'][i]['summaries'].update(data_bart['paragraphs'][i]['summaries'])
				data['paragraphs'][i]['summaries'].update(data_pysum['paragraphs'][i]['summaries'])
				data['paragraphs'][i]['summaries'].update(data_kw['paragraphs'][i]['summaries'])

			# Save modifications to preproc json files
			json.dump(data, open(PREPROC_PATH + d_id + PREPROC_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False,
			          indent=1)

merge_summaries()
