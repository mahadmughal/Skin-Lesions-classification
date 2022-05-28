from torch.utils.data import Dataset


class NewsClassificationDataset(Dataset):
  def __init__(self, text, labels):
    self.text = text
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    x = self.text[idx]
    y = self.labels[idx]

    return x, y
