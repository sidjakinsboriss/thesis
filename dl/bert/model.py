import torch
import transformers


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                                   num_labels=5,
                                                                                   problem_type='multi_label_classification')

    def forward(self, ids, mask):
        output = self.l1(ids, attention_mask=mask, return_dict=False)
        return output[0]  # Logits
