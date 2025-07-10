import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear

class TopKGate(nn.Module):
    """Red de puerta para seleccionar los k mejores expertos con cuantización BitLinear"""
    
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.k = k
        # Usar BitLinear para calcular logits de cada experto
        self.gate_linear = BitLinear(input_dim, num_experts, bias=False)

    def forward(self, x):
        # x shape: [batch_size * seq_len, input_dim]
        # logits shape: [batch_size * seq_len, num_experts]
        logits = self.gate_linear(x)

        # Seleccionar los k mejores expertos
        # top_k_logits shape: [batch_size * seq_len, k]
        # top_k_indices shape: [batch_size * seq_len, k]
        top_k_logits, top_k_indices = torch.topk(
            logits, self.k, dim=-1
        )

        # Aplicar softmax a los logits de los k mejores expertos
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Crear matriz de pesos completa para mantener dimensiones coherentes
        full_weights = torch.zeros_like(logits)
        full_weights.scatter_(1, top_k_indices, top_k_weights)

        return full_weights, top_k_indices




