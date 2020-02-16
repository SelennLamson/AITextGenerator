

#####################################################
# My file to do some tests, please do not modify ;) #
#####################################################

from src.third_party.BERT_NER.bert import Ner
from src.dataset_generation.ent_sum_preprocessing import *


# prepare_json_templates(True)


# Loading pre-trained model by feeding the folder where the pre-trained parameters are located.
model = Ner("../models/entity_recognition/BERT_NER_Large/")
perform_ner_on_all(model)

