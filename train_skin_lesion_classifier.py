from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
import pdb

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names):
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    
    df_cm = pd.DataFrame(cm, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()


def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]


def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):

    model.train()

    total_loss_train = []
    total_acc_train = []
    recall_scores = []
    train_loss = 0.0
    train_acc = 0.0


    for i, (inputs, labels) in enumerate(loader):
        images, labels = inputs.to(device), labels.to(device)
        N = images.size(0)

        optimizer.zero_grad()

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images.float())

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]

        recall_scores.append(recall_score(np.array(prediction.flatten()), np.array(labels), average='weighted', zero_division=1))
        
        train_acc = prediction.eq(labels.view_as(prediction)).sum().item()/N
        train_loss = loss.item()

        print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
            epoch, i + 1, len(loader), train_loss, train_acc))
        total_loss_train.append(train_loss)
        total_acc_train.append(train_acc)

    return {'Loss/train': total_loss_train, 'Accuracy/train': total_acc_train, 'UAR/train': recall_scores}, model


def val_epoch(epoch, model, criterion, loader, num_classes, device):

    model.eval()
    total_loss_val = []
    total_acc_val = []
    recall_scores = []
    val_loss = 0.0
    val_acc = 0.0

    for i, (inputs, labels) in enumerate(loader):
        images, labels = inputs.to(device), labels.to(device)
        N = images.size(0)

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images.float())
        prediction = outputs.max(1, keepdim=True)[1]

        val_acc = prediction.eq(labels.view_as(prediction)).sum().item()/N
        val_loss = criterion(outputs, labels).item()

        total_acc_val.append(val_acc)
        total_loss_val.append(val_acc)

        recall_scores.append(recall_score(np.array(prediction.flatten()), np.array(labels), average='weighted', zero_division=1))

    metrics_dict = {'Loss/val': total_loss_val, 'Accuracy/val': total_acc_val, 'UAR/val': recall_scores}

    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images.float())
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    confusion_mtx = confusion_matrix(y_label, y_predict)

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss, val_acc))
    print('------------------------------------------------------------')
    return metrics_dict, confusion_mtx


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
                
    num_classes = len(class_names)
    model.to(device)
    
    # Initialise Weights and Biases (wandb) project
    if ident_str is None:
      ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name)

    try:
        # Train by iterating over epochs
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict, net = train_epoch(epoch, model, optimizer, criterion,
                    train_loader, num_classes, device)
            model = net.train()
            val_metrics_dict, confusion_mtx = val_epoch(epoch, model, criterion, 
                    val_loader, num_classes, device)

            wandb.log({**train_metrics_dict, **val_metrics_dict})

    finally:
        run.finish()

    plot_confusion_matrix(confusion_mtx, class_names)
    torch.save(model, 'saved_models/news_classifier.pt')
