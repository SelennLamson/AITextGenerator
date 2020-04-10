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

novel_text = ' I have seen him wringing his hands after such a rebuff, and I am sure the annoyance and the terror he lived in must have greatly hastened his early and unhappy death. All the time he lived with us the captain made no change whatever in his dress but to buy some stockings from a hawker. One of the cocks of his hat having fallen down, he let it hang from that day forth, though it was a great annoyance when it blew. I remember the appearance of his coat, which he patched himself upstairs in his room, and which, before the end, was nothing but patches. He never wrote or received a letter, and he never spoke with any but the neighbours, and with these, for the most part, only when drunk on rum. The great sea-chest none of us had ever seen open. He was only once crossed, and that was towards the end, when my poor father was far gone in a decline that took him off. Dr. Livesey came late one afternoon to see the patient, took a bit of dinner from my mother, and went into the parlour to smoke a pipe until his horse should come down from the hamlet, for we had no stabling at the old Benbow. I followed him in, and I remember observing the contrast the neat, bright doctor, with his powder as white as snow and his bright, black eyes and pleasant manners, made with the coltish country folk, and above all, with that filthy, heavy, bleared scarecrow of a pirate of ours, sitting, far gone in rum, with his arms on the table.'

sentence = """Facebook CEO Mark Zuckerberg, left, makes the keynote speech at F8, the Facebook's developer conference, Tuesday, April 30, 2019, in San Jose, Calif. (AP Photo/Tony Avelar )
Facebook says that, unlike its past, its future is privacy
A trader works ahead of the closing bell on the floor of the New York Stock Exchange (NYSE) on April 12, 2019 in New York City. (Photo by Johannes EISELE / AFP)        (Photo credit should read JOHANNES EISELE/AFP/Getty Images)
Resilience is still the word for stocks"""



# KEYWORDS EXTRACTION
from gensim.summarization import keywords
model = keywords
' - '.join(model(big_text, lemmatize=False, pos_filter=('NN', 'JJ', 'VB')).split('\n'))

start = time.time()
keywords_sum = model(big_text, lemmatize=True, pos_filter=('NN', 'JJ', 'VB')).split('\n')
keywords_sum = [' - '.join(keywords_sum)]
end = time.time()
print(end-start)

model(big_text, lemmatize=True, pos_filter=('NN', 'JJ', 'VB'))



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

# BART
from transformers import pipeline
summariser = pipeline('summarization')
match_summary = summariser(big_text, min_length=5, max_length=50)
print(match_summary[0]['summary_text'])




### PYSUM
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# Text
document = big_text

# Init
auto_abstractor = AutoAbstractor() # Object of automatic summarization.
auto_abstractor.tokenizable_doc = SimpleTokenizer()  # Set tokenizer.
auto_abstractor.delimiter_list = ['.','\n']  # Set delimiter for making a list of sentence.
abstractable_doc = TopNRankAbstractor()  # Object of abstracting and filtering document.

# Summarize document.
result_dict = auto_abstractor.summarize(text, abstractable_doc)
#summary = result_dict['summarize_result'][max(result_dict['scoring_data'], key=lambda x:x[1])[0]]

max = 0
for i, item in enumerate(result_dict['scoring_data']):
	if item[1] > max:
		id = i
		max = item[1]
summary = [''.join(result_dict['summarize_result'][id])]




###  Pointer Generator Network

from eazymind.nlp.eazysum import Summarizer
key = 'a6bd47ccfdc8f0ffc5b904cd7d90f840'
sum = Summarizer(key)

text = big_text
# inputs = inputs.replace('“', '').replace('”', '').replace('’', '')
input = text.encode('Latin-1', 'ignore')
input = input.decode('ascii')
res = print(sum.run(input))
print(res)


from eazymind.nlp.eazysum import Summarizer
key = 'a6bd47ccfdc8f0ffc5b904cd7d90f840'

sentence = """Facebook CEO Mark Zuckerberg, left, makes the keynote speech at F8, the Facebook's developer conference, Tuesday, April 30, 2019, in San Jose, Calif. (AP Photo/Tony Avelar )
Facebook says that, unlike its past, its future is privacy
A trader works ahead of the closing bell on the floor of the New York Stock Exchange (NYSE) on April 12, 2019 in New York City. (Photo by Johannes EISELE / AFP)        (Photo credit should read JOHANNES EISELE/AFP/Getty Images)
Resilience is still the word for stocks"""

summarizer = Summarizer(key)
print(summarizer.run(sentence))



### T5

from transformers import T5Tokenizer, TFT5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
input_ids = tokenizer.encode(text, return_tensors="tf")  # Batch size 1
summary_ids = model.generate(input_ids)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

from transformers import T5Tokenizer, TFT5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="tf")  # Batch size 1
outputs = model(input_ids, decoder_input_ids=input_ids)
prediction_scores = outputs[0]
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my friends', return_tensors='tf')
greedy_output = model.generate(input_ids)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
