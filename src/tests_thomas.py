

#####################################################
# My file to do some tests, please do not modify ;) #
#####################################################

# from src.third_party.BERT_NER.bert import Ner
# from src.dataset_generation.ent_sum_preprocessing import *
# from src.dataset_generation.paragraph_preprocessing import *
import transformers as ppb
import json
import numpy as np
import torch

# separate_in_paragraphs()

# prepare_json_templates(True)
#
# # Loading pre-trained model by feeding the folder where the pre-trained parameters are located.
# model = Ner("../models/entity_recognition/BERT_NER_Large/")
# perform_ner_on_all(model)
#
#
#

file_id = "517"

# Retrieving BERT model and weights handles
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Loading data file
data = json.load(open('../data/ent_sum/' + file_id + '_entsum.json', 'r'))
paragraphs = data['novel']['paragraphs']

# Extracting texts of each paragraph
texts = [p['text'] for p in paragraphs]

# Preparing sequences in batch
tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in texts]
max_len = max(len(t) for t in tokenized)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])

# Building attention mask to hide padding
attention_mask_np = np.where(padded != 0, 1, 0)

# Embedding the sentences with BERT
batch_size = 100
model.cuda()
embeddings = np.zeros((len(padded), 768))
start_i = 0
batch_i = 0
while start_i < len(padded):
	input_ids = torch.LongTensor(padded[start_i:start_i + batch_size]).cuda()
	attention_mask = torch.tensor(attention_mask_np[start_i:start_i + batch_size]).cuda()
	with torch.no_grad():
		last_hidden_states = model.forward(input_ids, attention_mask=attention_mask)
	print("Batch", batch_i, "done.")
	embeddings[start_i:start_i + batch_size, :] = last_hidden_states[0][:,0,:].cpu().numpy()
	start_i += batch_size
	batch_i += 1
print("BERT task finished.")

np.save('../data/embeddings/' + file_id + '_paragraphs_embed.npy', embeddings)


