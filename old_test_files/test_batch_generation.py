from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

if __name__ == "__main__":
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')

    input_ids = tokenizer.encode('Mirror, mirror on the wall, who\'s the fairest of them all? Alex, Gael or Thomas ?',
                                 return_tensors='pt')

    print(input_ids)
    mask = (input_ids != tokenizer.pad_token_id).long()

    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=mask
        )

    print("Output \n" + 100 * '-')
    print("mask :", mask)
    print(sample_output[0])
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
