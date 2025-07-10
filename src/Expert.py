import torch
import torch.nn as nn
from .bitlinear import BitLinear

class Expert(nn.Module):
    """Red Experta Feed-Forward para MoE con cuantización BitLinear"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Usar BitLinear en lugar de nn.Linear para cuantización
        self.fc1 = BitLinear(input_dim, hidden_dim)
        self.fc2 = BitLinear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Aplicar primera capa BitLinear
        x = self.fc1(x)
        # Aplicar función de activación
        x = self.activation(x)
        # Aplicar segunda capa BitLinear
        x = self.fc2(x)
        return x
