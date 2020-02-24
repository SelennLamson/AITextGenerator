

#####################################################
# My file to do some tests, please do not modify ;) #
#####################################################


from src.dataset_generation.ent_sum_preprocessing import perform_ner_on_file

from src.flexible_models import FlexibleBERTNER

model = FlexibleBERTNER("../models/entity_recognition/BERT_NER_Large/", batch_size=128, max_length=2000)
perform_ner_on_file(model)




# prepare_json_templates(True)
#
# # Loading pre-trained model by feeding the folder where the pre-trained parameters are located.
# model = Ner("../models/entity_recognition/BERT_NER_Large/")
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


