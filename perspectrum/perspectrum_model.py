import torch
from typing import List
from transformers import (WEIGHTS_NAME, BertConfig,
                         BertForSequenceClassification, BertTokenizer,
                         RobertaConfig,
                         RobertaForSequenceClassification,
                         RobertaTokenizer,
                         InputExample)
from transformers import glue_convert_examples_to_features as convert_examples_to_features


class PerspectrumTransformerModel:
    def __init__(self, model_type, model_path, model_name="roberta-large", cuda=True, **kwargs):
        """
        Load pretrained model
        """
        self.device = "cuda" if cuda else "cpu"

        if model_type == "roberta":
            self.model_type = "roberta"
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        else:
            self.model_type = "bert"
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)

    def predict_batch(self, sent_pairs, label_set=(0, 1), max_sequnce_length=128, batch_size=20) -> List:
        """
        Run prediction
        :param sent_pairs: a list of sentence pairs to predict
        :param label_set: set of labels
        :param max_sequnce_length
        :param batch_size
        :return: a list of
        """
        predictions = []
        for i in range(0, len(sent_pairs), batch_size):
            examples = []
            sent_pair_batch = sent_pairs[i:i+batch_size]
            for sent_pair in sent_pair_batch:
                examples.append(
                    InputExample(guid="dummy", text_a=sent_pair[0], text_b=sent_pair[1], label=label_set[0]))

            features = convert_examples_to_features(examples,
                                                    self.tokenizer,
                                                    label_list=label_set,
                                                    max_length=max_sequnce_length,
                                                    output_mode="classification",
                                                    )

            batch_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            batch_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            batch_labels = torch.tensor([f.label for f in features], dtype=torch.long)

            batch_input_ids = batch_input_ids.to(self.device)
            batch_attention_mask = batch_attention_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)

            if self.model_type == "bert":
                batch_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
                batch_token_type_ids = batch_token_type_ids.to(self.device)
            else:
                batch_token_type_ids = None

            inputs = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
                "token_type_ids": batch_token_type_ids
            }

            with torch.no_grad():
                output = self.model(**inputs)
                tmp_eval_loss, logits = output[:2]
                logits = logits.detach().cpu().numpy()
                predictions.extend(logits[:, 1])

        return predictions


if __name__ == '__main__':
    import sys
    model = PerspectrumTransformerModel("roberta", sys.argv[1], cuda=True)
    print(model.predict([("123", "123"), ("asd", "asd")]))