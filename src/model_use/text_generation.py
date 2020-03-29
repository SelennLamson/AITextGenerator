from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    input_ids = tokenizer.encode('Mirror, mirror on the wall, who\'s the fairest of them all? Alex, Gael or Thomas ?',
                                 return_tensors='pt')

    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_p=0.92,
        top_k=0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
