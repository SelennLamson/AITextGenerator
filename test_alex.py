big_text = """Harry Potter’ is the most miserable, lonely boy you can imagine. 
           He’s shunned by his relatives, the Dursley’s, that have raised him since he was an infant. 
           He’s forced to ’live in the cupboard under the stairs, forced to wear his cousin Dudley’s 
           hand-me-down clothes, and forced to go to his neighbour’s house when the rest of the family 
           is doing something fun. Yes, he’s just about as miserable as you can get.
           Harry’s world gets turned upside down on his 11th birthday, however. A giant, Hagrid, informs Harry that he’s really 
           a wizard, and will soon be attending Hogwarts School of Witchcraft and Wizardry. Harry also learns that, 
           in the wizarding world, he’s a hero. When he was an infant, the evil Lord Voldemort killed his parents and 
           then tried to kill Harry too. What’s so amazing to everyone is that Harry survived, and allegedly destroyed Voldemort 
           in the process."""

# BART
from transformers import pipeline
summariser = pipeline('summarization')
match_summary = summariser(big_text, min_length=5, max_length=50)
print(match_summary[0]['summary_text'])


###  BART for summarisation
# General Init
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
model.eval()
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
# Tokenise text studied
ARTICLE_TO_SUMMARIZE = big_text
inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=40, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


# Pointer Generator Network
from eazymind.nlp.eazysum import Summarizer
key = 'a6bd47ccfdc8f0ffc5b904cd7d90f840'
try:
	sum = Summarizer(key)
except:
print(sum.run(text))

big_text = str(big_text)
text = big_text.encode('Latin-1', 'ignore')
text = text.decode('utf-8', 'ignore')

# Import libraries
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# Text
document = big_text

# Init
auto_abstractor = AutoAbstractor() # Object of automatic summarization.
auto_abstractor.tokenizable_doc = SimpleTokenizer()  # Set tokenizer.
auto_abstractor.delimiter_list = [".", "\n"]  # Set delimiter for making a list of sentence.
abstractable_doc = TopNRankAbstractor()  # Object of abstracting and filtering document.

# Summarize document.
result_dict = auto_abstractor.summarize(document, abstractable_doc)
#summary = result_dict['summarize_result'][max(result_dict['scoring_data'], key=lambda x:x[1])[0]]

max = 0
id = []
for i, item in enumerate(result_dict['scoring_data']):
	if item[1] > max:
		id.append(i)
		max = item[1]
	if item[1] == max and len(id) < 2:
		id.append(i)
summary = ''.join([result_dict['summarize_result'][j] for j in id])
