from torch.utils.data import DataLoader
import pandas as pd

from src.torch_loader import DatasetFromRepo, VectorizeParagraph
from src.model_evaluation.metrics import entities_iou
from src.flexible_models import FlexibleBERTEmbed
from src.utils import ENTITY_TAGS

class Evaluation:
    """
    The Evaluation class will be used to evaluate a GPT2 model
    Roughly, the idea is to take as input :
        - a fine-tuned GPT2 model
        - a path to a folder contained preprocessed novel on which we want to evaluate the GPT2 model
        - a decoding strategy

    Then, the compute_metrics method will compute and return as a panda.dataframe the different metrics
    on each example on the dataset
    """
    def __init__(self, GPT_model, GPT_tokenizer, BERT_NER_model, path_to_repo, batch_size, decoding_strategy):
        """
        :param GPT_model: fine-tune GPT2 model to evaluate
        :param GPT_tokenizer: GPT2 tokenizer used from training time
        :param BERT_NER_model: FlexibleBERTNER model to detect entities for IOU metrics
        :param path_to_repo: path to repo containing the novel as json file on which we want to evaluate the model
        :param batch_size: batch size that will be use for prediction
        :param decoding_strategy: dict containing the paramater for the transformers.generate method
        """
        self.GPT_model = GPT_model
        self.BERT_ner_model = BERT_NER_model
        self.GPT_tokenizer = GPT_tokenizer
        self.BERT_sim_model = FlexibleBERTEmbed
        vectorize_paragraph = VectorizeParagraph(GPT_tokenizer, block_size=1020, train_mode=False)
        self.dataset = DatasetFromRepo(path=path_to_repo, transform=vectorize_paragraph)
        self.batch_size = batch_size
        self.decoding_strategy = decoding_strategy
        self.max_length = decoding_strategy['max_length']

    def prediction(self, input_ids):
        """
        :param input_ids: torch.tensors of shape (batch_size, max_length)
        :return: list[str] : list of generated text for each input
            IMPORTANT : It return the text generated after the input
        """
        mask = (input_ids != self.GPT_tokenizer.pad_token_id).long()
        self.decoding_strategy['max_length'] = self.max_length + input_ids.shape[1]
        print(self.decoding_strategy)
        outputs_id = self.GPT_model.generate(input_ids=input_ids,
                                             pad_token_id=self.GPT_tokenizer.eos_token_id,
                                             attention_mask=mask,
                                             **self.decoding_strategy)

        # only keep the token corresponding to the generation part
        # this is because transformers.generate methods also return the input part
        truncated_outputs_id = outputs_id[:, input_ids.shape[1]:]

        return [self.GPT_tokenizer.decode(truncated_outputs_id[i], skip_special_tokens=True)
                for i in range(outputs_id.shape[0])]

    def pad_left_side(self, sequences):
        """
        Modification of torch.nn.utils.rnn.pad_sequence so that we pad left side and not right side
        :param sequences : list of tensors
        :return tensor of shape (len(sequences), max_length of sequence in sequences)
                the tensor are padded on the left side using pad_token_id from GPT2 tokenizer
        """
        max_len = max([s.size(0) for s in sequences])
        out_dims = (len(sequences), max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(self.GPT_tokenizer.pad_token_id)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, max_len - length:] = tensor
        return out_tensor

    def custom_collate(self, data_samples):
        """
        Will be use to instanciate a torch dataloader from our custom dataset
        :param data_samples: list of sample
            each sample is a tupple ([torch.tensor] input_ids, [str] true_P2)
        :return: a tupple: (concatenation on batch axis using left padding of the input_ids,
                            list of true_P2)
        """
        input_ids = self.pad_left_side([sample[0] for sample in data_samples])
        target_paragraphs = [sample[1] for sample in data_samples]

        return input_ids, target_paragraphs

    def compute_metrics(self, nb_examples_to_evaluate=None):
        """
        Compute the following metrics on each example construction from the evaluation repo :
        - bert_similarity between each predicted P2 and true P2
        - entities IOU on each classes (ENT, ORG, LOC, MISC) between each predicted P2 and true P2
        :param : nb_examples_to_evaluate, if specify we only compute the metrics for nb_examples from the dataset
                  (+/- batch_size)
        :return: pd.DataFrame:
            - one row for each paragraph
            - one column for each metrics
        """
        dataloader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                collate_fn=self.custom_collate)

        metrics = pd.DataFrame(columns=["True_P2", "Pred_P2", "PER_iou", "ORG_iou", "LOC_iou", "MISC_iou"])

        self.GPT_model.eval()
        idx = 0
        for input_ids, true_P2 in dataloader:
            pred_P2 = self.prediction(input_ids)

            iou = entities_iou(true_paragraphs=true_P2,
                               pred_paragraphs=pred_P2,
                               ner_model=self.BERT_ner_model,
                               class_tags=list(ENTITY_TAGS))  # just to check if it work for now

            metrics = metrics.append(pd.concat([pd.DataFrame(true_P2, columns=['True_P2']),
                                                pd.DataFrame(pred_P2, columns=['Pred_P2']),
                                                pd.DataFrame(iou['PER'], columns=["PER_iou"]),
                                                pd.DataFrame(iou['ORG'], columns=["ORG_iou"]),
                                                pd.DataFrame(iou['LOC'], columns=["LOC_iou"]),
                                                pd.DataFrame(iou['MISC'], columns=["MISC_iou"])],
                                               axis=1),
                                     ignore_index=True)

            idx += input_ids.shape[0]
            if nb_examples_to_evaluate is not None and idx >= nb_examples_to_evaluate:
                break

        return metrics
