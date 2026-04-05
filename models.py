# models.py — network architectures

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Standard CNN for MNIST classification.
    Used as the baseline model and as the teacher in defensive distillation.

    Returns raw logits — apply log_softmax externally.
    """
    def __init__(self):
        super().__init__()
        self.conv1    = nn.Conv2d(1, 32, 3, 1)
        self.conv2    = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1      = nn.Linear(9216, 128)
        self.fc2      = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)           # raw logits


class NetF1(nn.Module):
    """
    Smaller student network used in defensive distillation.
    Half the filters, smaller FC layer — cheaper to train.

    Returns raw logits — apply log_softmax externally.
    """
    def __init__(self):
        super().__init__()
        self.conv1    = nn.Conv2d(1, 16, 3, 1)
        self.conv2    = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1      = nn.Linear(4608, 64)
        self.fc2      = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)           # raw logits
