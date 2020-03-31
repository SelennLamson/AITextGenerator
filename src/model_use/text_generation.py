from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.model_training.update_model import add_special_tokens

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'left'
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    add_special_tokens(model=model, tokenizer=tokenizer)
    input_ids = tokenizer.encode('Mirror, mirror on the wall, who\'s the fairest of them all? Alex, Gael or Thomas ?',
                                 return_tensors='pt')
    print(input_ids)

    sample_output = model.generate(
        input_ids,
        max_lenght=50,
        do_sample=True,
        top_p=0.92,
        top_k=0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
