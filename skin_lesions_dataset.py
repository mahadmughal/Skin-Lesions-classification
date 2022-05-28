import os
import pandas as pd
import torch
from torchvision.io import read_image
import pdb


class SkinLesionsDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, labels_fname, transform=None):
    self.labels_fname = pd.read_csv(labels_fname)
    self.img_dir = img_dir
    self.transform = transform

  def __len__(self):
    return len(self.labels_fname)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, f'{self.labels_fname.iloc[idx, 1]}.jpg')
    image = read_image(img_path)
    label = self.labels_fname.iloc[idx, 2]

    if self.transform:
      image = self.transform(image)

    return image, label
