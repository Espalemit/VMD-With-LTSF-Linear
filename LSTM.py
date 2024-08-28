import pandas as pd
import numpy as np
import matplotlib as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Multivariate Multivariate Multivariate Multivariate
# LSTM Class
class MVLSTM(nn.Module):
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def __init__(self, feat_lenn, inp_size, hid_size, num_stack_layer):
    super().__init__()
    self.hid_size = hid_size
    self.feat_len = feat_lenn
    self.num_stack_layer = num_stack_layer
    self.lstm = nn.LSTM(inp_size, hid_size, num_stack_layer, batch_first=True)
    self.fc = nn.Linear(hid_size, feat_lenn)

  def forward(self, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stack_layer, batch_size, self.hid_size).to(device)
    c0 = torch.zeros(self.num_stack_layer, batch_size, self.hid_size).to(device)
    out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
    out = self.fc(out[:, -1, :])
    return out

#Univariate Univariate Univariate Univariate
# LSTM Class
class LSTM(nn.Module):
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def __init__(self, inp_size, hid_size, num_stack_layer):
    super().__init__()
    self.hid_size = hid_size
    self.num_stack_layer = num_stack_layer
    self.lstm = nn.LSTM(inp_size, hid_size, num_stack_layer, batch_first=True)
    self.fc = nn.Linear(hid_size, 1)

  def forward(self, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stack_layer, batch_size, self.hid_size).to(device)
    c0 = torch.zeros(self.num_stack_layer, batch_size, self.hid_size).to(device)
    out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
    out = self.fc(out[:, -1, :])
    return out
