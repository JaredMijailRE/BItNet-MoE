import torch
import torch.nn as nn

class Expert(nn.Module):
    """Feed-forward Expert Network"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 == nn.Linear(input_dim, hidden_dim)
        self.fc2 == nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
