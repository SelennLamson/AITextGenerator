from torch.utils.data import DataLoader
from src.torch_loader import DatasetFromRepo, VectorizeParagraph


class Evaluation:
    def __init__(self, model, tokenizer, path_to_repo, batch_size, decoding_strategy):
        """
        :param model: fine-tune GPT2 model to evaluate
        :param tokenizer: GPT2 tokenizer used from training time
        :param path_to_repo: path to repo containing the novel as json file on which we want to evaluate the model
        :param batch_size: batch size that will be use for prediction
        :param decoding_strategy: dict containing the paramater for the transforms.generate
        """
        self.model = model
        self.tokenizer = tokenizer
        vectorize_paragraph = VectorizeParagraph(tokenizer, block_size=1020, train_mode=False)
        self.dataset = DatasetFromRepo(path=path_to_repo, transform=vectorize_paragraph)
        self.batch_size = batch_size
        self.decoding_strategy = decoding_strategy

    def prediction(self, input_ids):
        """
        :param input_ids: torch.tensors of shape (batch_size, max_length)
        :return: list[str] : list of generated text for each input
        """
        mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs_id = self.model.generate(input_ids, do_sample=True, attention_mask=mask, **self.decoding_strategy)

        return [self.tokenizer.decode(outputs_id[i], skip_special_tokens=True) for i in range(outputs_id.shape[0])]

    @staticmethod
    def collate_fn(data_samples):

    def compute_metrics(self, nb_examples, decoding_strategy=None, batch_size=1):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        for input_ids, true_P2 in dataloader:
            pred_P2 = self.prediction(input_ids)

