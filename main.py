from src.json_generation import paragraph_preprocessing
from src.json_generation import ent_sum_preprocessing
from src.json_generation import data_preprocessing
from src.flexible_models import FlexibleBERTNER
from src.flexible_models import FlexibleBERTSum
from src.utils import *

# Populate the cash metadata first - long operations
# cache = get_metadata_cache()
# cache.populate()

# Extract a book (text and metadata) 
# create_file = data_preprocessing.DataPrepro()
# create_file.create_json(b_id = 1342)

# Split into paragraphs
# paragraph_preprocessing.separate_paragraphs_all_files(overwrite=False)
# parser = paragraph_preprocessing.ParagraphParser(min_threshold=20, min_length=600, max_length=900)
# paragraph_preprocessing.separate_in_paragraphs(parser, d_id='1342')

# Create ent_sum template
# ent_sum_preprocessing.prepare_json_templates(True)

# Perform NER
# model = FlexibleBERTNER(BERT_NER_LARGE, batch_size=128, max_length=2000)
# ent_sum_preprocessing.perform_ner_on_file(model)
# ent_sum_preprocessing.perform_ner_on_file(model, d_id= '1342')

# Summarise
model_sum = FlexibleBERTSum()
# bert_sum, bart = model_sum.predict()
ent_sum_preprocessing.add_summaries([model_sum], replace=True, d_id='135')

