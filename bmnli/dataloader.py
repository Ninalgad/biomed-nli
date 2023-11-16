import torch
from torch.utils.data import DataLoader


class StaticDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def get_static_loader(tokenizer, x, y=None, max_length=512, batch_size=8, shuffle=False):
    encodings = tokenizer(list(x), padding="max_length", truncation=True,
                          max_length=max_length)
    loader = DataLoader(StaticDataset(encodings, y),
                        batch_size=batch_size, shuffle=shuffle)
    return loader
