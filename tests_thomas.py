

############################################
# THOMAS's TEST FILE, PLEASE DO NOT MODIFY #
############################################


from src.json_generation.ent_sum_preprocessing import perform_ner_on_all
from src.flexible_models import FlexibleBERTNER
from src.utils import *

model = FlexibleBERTNER(BERT_NER_LARGE, batch_size=256, max_length=128)
perform_ner_on_all(model)




# prepare_json_templates(True)
#
# # Loading pre-trained model by feeding the folder where the pre-trained parameters are located.
# model = Ner("BERT_NER_LARGE")
# perform_ner_on_all(model)
#
#
#
# from src.flexible_models import FlexibleBERTEmbed
#
# import json
# import numpy as np
# file_id = "517"
#
# # Loading data file
# data = json.load(open('../data/ent_sum/' + file_id + '_entsum.json', 'r'))
# paragraphs = data['novel']['paragraphs']
#
# print(len(paragraphs))
#
# # Extracting texts of each paragraph
# texts = [p['text'] for p in paragraphs]
#
# embedder = FlexibleBERTEmbed(max_length=2000, batch_size=50, cuda_device=0)
# embeddings = embedder(texts)
#
# print(embeddings.shape)
#
# print("BERT task finished.")

# np.save('../data/embeddings/' + file_id + '_paragraphs_embed.npy', embeddings)


