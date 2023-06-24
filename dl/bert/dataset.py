import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BertDataset(Dataset):
    def __init__(self, text, labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx] if isinstance(self.text[idx], str) else ''

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': self.labels[idx]
        }
