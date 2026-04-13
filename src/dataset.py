import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class AnimeDataset(Dataset):
    def __init__(self, synopses, labels, max_length=256):
        self.synopses = synopses
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.synopses)

    def __getitem__(self, idx):
        synopsis = self.synopses[idx]
        label = self.labels[idx]

        encoding = tokenizer(
            synopsis,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }
    