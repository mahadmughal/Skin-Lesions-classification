import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import collections
import skin_lesions_dataset
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import train_skin_lesion_classifier
from custom_skin_lesion_network import custom_model
from custom_skin_lesion_network import CustomSkinLesionNetwork
import pdb

IMG_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
LABELS_MAP = { 0: "MEL", 1: "NV", 2: "BCC", 3: "AKIEC", 4: "BKL", 5: "DF", 6: "VASC" }

TRAIN_LABELS_PATH = 'dataset/img/train.csv'
VAL_LABELS_PATH = 'dataset/img/val.csv'
IMAGES_PATH = 'dataset/img'
BATCH_SIZE = 64
num_epochs=10
input_size = 224
num_of_classes = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_loading():
  train_df = pd.read_csv(TRAIN_LABELS_PATH)
  val_df = pd.read_csv(VAL_LABELS_PATH)

  return train_df, val_df


def one_hot_to_integers(train_df, val_df):
  train_labels = np.argmax(train_df.iloc[:, 1:].values, axis=1)
  val_labels = np.argmax(val_df.iloc[:, 1:].values, axis=1)

  train_df['label'] = train_labels
  val_df['label'] = val_labels

  train_image_label_map = train_df[['image', 'label']]
  val_image_label_map = val_df[['image', 'label']]

  train_image_label_map.to_csv('train_image_label.csv')
  val_image_label_map.to_csv('val_image_label.csv')

  return train_labels, val_labels 


def plot_label_distribution(labels, split):
  elements_count = collections.Counter(labels)

  sorted_counts = collections.OrderedDict(sorted(elements_count.items()))

  # for key, value in elements_count.items():
  #   print(f"{key}: {value}")

  # Plot a histogram of the distribution
  plt.title(f'{split} distribution')
  plt.bar(IMG_CLASS_NAMES, sorted_counts.values())
  plt.xlabel('Class')
  plt.ylabel('Num examples')
  plt.show()


def data_tranformation():
  train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)])
  val_transform = transforms.Compose([transforms.Resize((input_size,input_size))])

  return train_transform, val_transform


def prepare_dataset():
  train_transform, val_transform = data_tranformation()

  training_data = skin_lesions_dataset.SkinLesionsDataset(
    IMAGES_PATH, 'train_image_label.csv', transform=train_transform)
  test_data = skin_lesions_dataset.SkinLesionsDataset(
    IMAGES_PATH, 'val_image_label.csv', transform=val_transform)

  train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=4)
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4)

  # for X, y in test_dataloader:
  #   print(f"Shape of X [N, C, H, W]: {X.shape}")
  #   print(f"Shape of y: {y.shape} {y.dtype}")
  #   break

  return train_dataloader, test_dataloader


def prepare_pretrained_model(train_dataloader, test_dataloader):
  model_conv = models.resnet18(pretrained=True)

  for param in model_conv.parameters():
    param.requires_grad = False

  num_ftrs = model_conv.fc.in_features
  model_conv.fc = nn.Linear(num_ftrs, len(IMG_CLASS_NAMES))
  model_ft = model_conv.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

  model_ft = train_skin_lesion_classifier.train_model(
    model_ft, train_dataloader, test_dataloader, optimizer_ft, criterion,
    IMG_CLASS_NAMES, num_epochs, project_name="CSE5DL Assignment Task 1",
    ident_str= "fill me in here")


def prepare_custom_model(train_dataloader, test_dataloader):
  model = CustomSkinLesionNetwork(num_of_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

  model_ft = train_skin_lesion_classifier.train_model(
    model, train_dataloader, test_dataloader, optimizer_ft, criterion,
    IMG_CLASS_NAMES, num_epochs, project_name="CSE5DL Assignment Task 1",
    ident_str= "fill me in here")


train_df, val_df = data_loading()
train_labels, val_labels = one_hot_to_integers(train_df, val_df)
plot_label_distribution(train_labels, 'train')
plot_label_distribution(val_labels, 'validation')
train_dataloader, test_dataloader = prepare_dataset()
prepare_pretrained_model(train_dataloader, test_dataloader)
# prepare_custom_model(train_dataloader, test_dataloader)



