import torch
import torch.nn as nn
import torchvision
from transformers import AutoModelForSequenceClassification

# TODO Task 1c - Implement a SimpleBNConv


# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.

# TODO Task 1f - Create your own models

# TODO Task 2c - Complete the TextMLP class

class TextMLP(nn.Module):
    def __init__(self, vocab_size, sentence_len, hidden_size, n_classes=4):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size//2),
            nn.Flatten(),
            # To determine the input size of the following linear layer think 
            # about the number of words for each sentence and the size of each embedding. 
            ## nn.Linear(.... ,  hidden_size),
            #.....

        )


# TODO Task 2c - Create a model which uses a distilbert-base-uncased
#                by completing the following.
class DistilBertForClassification(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
    #   ....

