import torch
import torch.nn as nn
import torch.nn.functional as F
from .Expert import Expert
from .GatingNetwork import TopKGate

class MoELayer(nn.Module):
    """Capa de Mezcla de Expertos (Mixture of Experts)"""
    
    def __init__(self, input_dim, output_dim, num_experts, k=1, 
                 expert_hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.output_dim = output_dim

        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim * 4  # Práctica común en transformers

        # Red de puerta y expertos
        self.gate = TopKGate(input_dim, num_experts, k)
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim, output_dim) 
             for _ in range(num_experts)]
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        original_shape = x.shape
        batch_size, seq_len = original_shape[0], original_shape[1]
        
        # Aplanar para procesamiento: [batch_size * seq_len, input_dim]
        x_flat = x.view(-1, original_shape[-1])

        # Obtener pesos de puerta e índices de expertos
        gate_weights, top_k_indices = self.gate(x_flat)

        # Inicializar tensor de salida
        final_output = torch.zeros(
            x_flat.shape[0], self.output_dim, 
            device=x.device, dtype=x.dtype
        )
        
        # Procesar cada experto seleccionado
        for i in range(self.k):
            # Para cada posición k, obtener índices de expertos y pesos
            expert_indices = top_k_indices[:, i]  # [batch_size * seq_len]
            expert_weights = gate_weights.gather(1, expert_indices.unsqueeze(-1)).squeeze(-1)  # [batch_size * seq_len]
            
            # Procesar cada experto
            for expert_idx in range(self.num_experts):
                # Máscara para tokens asignados a este experto en esta posición k
                expert_mask = (expert_indices == expert_idx)
                
                if expert_mask.any():
                    # Obtener tokens para este experto
                    expert_tokens = x_flat[expert_mask]
                    
                    # Procesar tokens con el experto
                    expert_output = self.experts[expert_idx](expert_tokens)
                    
                    # Obtener pesos correspondientes
                    weights = expert_weights[expert_mask].unsqueeze(-1)
                    
                    # Aplicar pesos y acumular resultado
                    weighted_output = expert_output * weights
                    final_output[expert_mask] += weighted_output
            
        # Restaurar forma original: [batch_size, seq_len, output_dim]
        final_output = final_output.view(batch_size, seq_len, self.output_dim)
        
        return final_output, gate_weights