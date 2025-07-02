import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    """Gate module para seleccionar los expertos"""
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.k = k
        # Capa lineal para computar logits de expertos
        self.gate_linear = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        # x shape: [batch*size * seq_len, input_dim]
        # logits shape: [batch_size * seq_len, num_experts]
        logits = self.gate_linear(x)

        # Selecionamos los top-k expertos
        # top_k_logits shape: [batch_size * seq_len, k]
        # top_k_indices shape: [batch_size * seq_len, k]
        top_k_logits, top_k_indices = torch.topk(
            logits, self.k, dim=-1
        )

        # Aplicamos softmax a los top-k logits pesos
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Creamos una sparse weight matriz para que las dimensiones coincidan
        full_weights = torch.zeros_like(logits)
        full_weight.scatter_(1, top_k_indices, top_k_weights)

        return full_weight, top_k_indices




