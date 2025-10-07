"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        x = self.relu(torch.mm(adj, self.fc1(x_in)))
        x = self.dropout(x)
        x_hidden = self.relu(torch.mm(adj, self.fc2(x)))  # Output of second message-passing layer
        x = self.fc3(x_hidden)
        return F.log_softmax(x, dim=1), x_hidden # Task 13
    




