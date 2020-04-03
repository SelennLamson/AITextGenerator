from src.flexible_models import FlexibleGPT2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == '__main__':

    GPT2_model = FlexibleGPT2(
        model=GPT2LMHeadModel.from_pretrained('gpt2'),
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
        decoding_strategy={
            'max_length':50,
            'do_sample':True,
            'top_k':50,
            'top_p':0.95
        }
    )

    input_str = 'Mirror, mirror on the wall, who\'s the fairest of them all?'
    input_ids = GPT2_model.tokenizer.encode(input_str, return_tensors='pt')

    print(input_ids)
    predictions = GPT2_model.predict(input_ids, nb_samples=3)
    for i, prediction in enumerate(predictions):
        print("------------\nprediction n°", i+1, " for input :", input_str, "->", prediction)
