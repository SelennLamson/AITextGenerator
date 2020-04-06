big_text = "Harry Potter is the most miserable, lonely boy you can imagine.  \
            He’s shunned by his relatives, the Dursley’s, that have raised him since he was an infant.  \
            He’s forced to live in the cupboard under the stairs, forced to wear his cousin Dudley’s  \
            hand-me-down clothes, and forced to go to his neighbour’s house when the rest of the family  \
            is doing something fun. Yes, he’s just about as miserable as you can get. \
            Harry’s world gets turned upside down on his 11th birthday, however. A giant, Hagrid, informs Harry that he’s really \
			a wizard, and will soon be attending Hogwarts School of Witchcraft and Wizardry. Harry also learns that,  \
			in the wizarding world, he’s a hero. When he was an infant, the evil Lord Voldemort killed his parents and  \
			then tried to kill Harry too. What’s so amazing to everyone is that Harry survived, and allegedly destroyed Voldemort \
			in the process."

# BART
from transformers import pipeline
summariser = pipeline('summarization')
match_summary = summariser(big_text, min_length=5, max_length=50)
print(match_summary[0]['summary_text'])

# BART for summarisation
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
model.eval()
start = time.time()
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
ARTICLE_TO_SUMMARIZE = big_text
inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=40, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

