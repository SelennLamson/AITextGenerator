from transformers import GPT2Tokenizer

class TransformParagraphes:
    """
    class TransformParagraphes :
    from {input: (metadata, P1, P3), target: P2} generate {input: X, output: Y}
    when X and Y are vectorized token tensors obtains using GPT2Tokenizer

    An instance of this class will be callable on {input: (metadata, P1, P3), target: P2}
    Later it will be possible to configure it (for instance for randomly erase some data)
    """
    def __init__(self):
        """
        For now, we simply apply GPT2Tokenizer
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'begin_P1': '[P1]',
                                           'begin_P3': '[P3]',
                                           'small': '[S]',
                                           'medium': '[M]',
                                           'large': '[L]',
                                           'begin_theme': '[T]',
                                           'begin_summary': '[Sum]',
                                           'begin_person': '[Ent]'})
        # We can add locations / organisations later
        # IMPORTANT : BEFORE FINE-TUNING THE MODEL, WE WILL NEED TO RESIZE THE TOKEN EMBEDDINGS
        # model.resize_token_embeddings(len(tokenizer))

    def reduce_string_list(self, string_list):
        """
        :param string_list: list of string ["str1", "str2", ..., "strN"]
        :return: a string "str1 - str2 - .. - strN"
        will be used of entities list
        """
        if len(string_list) == 1:
            return string_list[0]
        else:
            return string_list[0] + "-" + self.reduce_string_list(string_list[1:])

    def bin_size(self, size):
        """
        :param size: int <-> size of a paragraph
        :return: special token [S], [M], [L] <-> corresponding binned size
        for now only return [M]
        """
        return '[M]'

    def __call__(self, sample):
        """
        For now, we simply use
        """

        metadata, P1, P3 = sample['input']
        P2 = sample['target']
        input_string = '[P1]' + P1['text'] + \
                       '[P3]' + P3['text'] + \
                       '[Sum]' + P2['summaries'] + \
                       '[T]' + metadata['theme'] + \
                       '[Ent]' + self.reduce_string_list(P2["persons"]) + \
                        self.bin_size(P2['size'])

        target_string = P2['text']

        return {'input':self.tokenizer.encode(input_string),
                'target':self.tokenizer.encode(target_string)}

