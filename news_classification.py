import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import manual_seed
manual_seed(16)
from random import seed
seed(16)

import pandas as pd
from news_classification_dataset import NewsClassificationDataset
import pdb
import collections
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import long, as_tensor, cat
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1
from torch import save, stack, no_grad
from torch.optim.lr_scheduler import CosineAnnealingLR
from random import random
from torch.optim import Adam
from custom_news_classification_model import CustomNewsClassificationModel
import train_news_classifier

TRAIN_DATASET = 'dataset/txt/train.csv'
VALIDATION_DATASET = 'dataset/txt/val.csv'
tokenizer = get_tokenizer('basic_english')

BATCH_SIZE = 64
LR = 0.0005
NUMBER_OF_CLASSES = 4
NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tec"]
NUMBER_OF_EPOCHS = 80

text_transf = lambda x: vocab(tokenizer(x))
label_transf = lambda x: int(x)

def data_loading():
  train_df = pd.read_csv(TRAIN_DATASET)
  val_df = pd.read_csv(VALIDATION_DATASET)

  return train_df, val_df


def prepare_data_file(train_df, val_df):
  train = [(' '.join([topic, text]), label-1) for label, topic, text in train_df.values]
  val = [(' '.join([topic, text]), label-1) for label, topic, text in val_df.values]

  train_x, train_y = [text for text, _ in train], [label for _, label in train]
  val_x, val_y = [text for text, _ in val], [label for _, label in val]

  train_ds = NewsClassificationDataset(train_x, train_y)
  val_ds = NewsClassificationDataset(val_x, val_y)

  return train, val, train_ds, val_ds


def plot_label_distribution(labels, split):
  elements_count = collections.Counter(labels)

  sorted_counts = collections.OrderedDict(sorted(elements_count.items()))

  for key, value in elements_count.items():
    print(f"{key}: {value}")

  # Plot a histogram of the distribution
  plt.title(f'{split} distribution')
  plt.bar(NEWS_CLASSES, sorted_counts.values())
  plt.xlabel('Class')
  plt.ylabel('Num examples')
  plt.show()


def tokenization(data):
  for text, _ in data:
    yield tokenizer(text)


def prepare_vocab(data):
  vocab = build_vocab_from_iterator(tokenization(data), specials=["<unk>"])
  vocab.set_default_index(vocab["<unk>"])
  # print(vocab(tokenizer('My name is Mauricio and I am an economist passionate of ML')))

  return vocab


def data_transformation(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_transf(_label))
        processed_text = as_tensor(text_transf(_text), dtype=long)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = as_tensor(label_list, dtype=long)
    offsets = as_tensor(offsets[:-1]).cumsum(dim=0)
    text_list = cat(text_list)
    return text_list, label_list, offsets


def prepare_data_loaders(data_ds):
  data_dl = DataLoader(
            data_ds, 
            batch_size=BATCH_SIZE, 
            collate_fn=data_transformation, 
            drop_last=True,
            num_workers=3
          )

  return data_dl


def train_custom_model(train_dataloader, test_dataloader, vocab):
  model = CustomNewsClassificationModel(
    len(vocab),
    num_class=NUMBER_OF_CLASSES
    ).to(device)
  optimizer = Adam(model.parameters(), lr=LR)
  criterion = torch.nn.NLLLoss()
  lr_scheduler = CosineAnnealingLR(optimizer, T_max = 2, eta_min = 1e-5)

  model = train_news_classifier.train_model(
    model, train_dataloader, test_dataloader, optimizer, criterion,
    NEWS_CLASSES, NUMBER_OF_EPOCHS, project_name="CSE5DL Assignment Task 1",
    ident_str= "fill me in here")



train_df, val_df = data_loading()
train, val, train_ds, val_ds = prepare_data_file(train_df, val_df)
plot_label_distribution([label for _, label in train], 'train')
plot_label_distribution([label for _, label in val], 'validation')
vocab = prepare_vocab(train)
train_dataloader = prepare_data_loaders(train_ds)
test_dataloader = prepare_data_loaders(val_ds)
train_custom_model(train_dataloader, test_dataloader, vocab)




