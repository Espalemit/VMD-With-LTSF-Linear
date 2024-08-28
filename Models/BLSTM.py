import pandas as pd
import numpy as np
import matplotlib as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# BLSTM Class
class MVBLSTM(nn.Module):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def __init__(self, num_classes, input_size, hidden_size, num_layers):
    super(MVBLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_size * 2, num_classes)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

class BLSTM(nn.Module):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(BLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_size * 2, num_classes)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out
