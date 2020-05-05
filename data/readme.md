Here are all the preprocessed files, at different stages:

1. **metadata/files/id.json** \
Contains the text files along with related book metadata, downloaded from Gutenberg.
2. **novel/id_novel.json** \
Contains the text files with NER performed on the whole novel, with positional indices for each entity.
3. **summaries/summarizer_id_preproc.json** \
Contains the text files with four different summaries of each paragraph
4. **preproc/id_preproc.json** \
Contains the text files splitted into paragraphs, where entities and summaries are associated to each paragraph.\

None of these files are stored in github, they can all be dowloaded on the data archive. \
See main ```readme.md``` file for the download link. \
See ```data.py``` for the full data pre-processing pipeline. 
