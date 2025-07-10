import torch
import torch.nn as nn
import torch.nn.functional as F
from .MoeLater import MoELayer
from .bitlinear import BitLinear

class TransformerBlockWithMoE(nn.Module):
    """Bloque Transformer con capa de Mezcla de Expertos (MoE)"""
    
    def __init__(self, embed_dim, num_heads, num_experts, k=1, 
                 dropout=0.1):
        super().__init__()
        # Mecanismo de atención multi-cabeza
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        # Capas de normalización
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Capa MoE que reemplaza la FFN tradicional
        self.moe_layer = MoELayer(embed_dim, embed_dim, num_experts, k)
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass del bloque transformer con MoE.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, embed_dim]
            mask: Máscara de atención opcional
            
        Returns:
            x: Salida del bloque [batch_size, seq_len, embed_dim]
            gate_weights: Pesos de puerta para pérdida auxiliar
        """
        # Bloque de atención multi-cabeza con conexión residual
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)  # Conexión residual
        x = self.norm1(x)
        
        # Bloque MoE con conexión residual
        moe_output, gate_weights = self.moe_layer(x)
        x = x + self.dropout(moe_output)  # Conexión residual
        x = self.norm2(x)
        
        # Devolver salida y pesos de puerta para pérdida auxiliar
        return x, gate_weights


class MoETransformer(nn.Module):
    """Transformer completo con capas MoE y cuantización BitLinear para entrenamiento"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                 num_experts, k=1, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embeddings de tokens y posición (mantener nn.Embedding para estos)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Stack de bloques transformer con MoE
        self.layers = nn.ModuleList([
            TransformerBlockWithMoE(embed_dim, num_heads, num_experts, k, dropout)
            for _ in range(num_layers)
        ])
        
        # Capa de salida con BitLinear para predicción de tokens
        self.output_projection = BitLinear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass del transformer MoE completo.
        
        Args:
            input_ids: IDs de tokens [batch_size, seq_len]
            attention_mask: Máscara de atención opcional
            
        Returns:
            logits: Logits de predicción [batch_size, seq_len, vocab_size]
            all_gate_weights: Lista de pesos de puerta de todas las capas
        """
        batch_size, seq_len = input_ids.shape
        
        # Crear embeddings posicionales
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Combinar embeddings de tokens y posición
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Aplicar todas las capas transformer
        all_gate_weights = []
        for layer in self.layers:
            x, gate_weights = layer(x, attention_mask)
            all_gate_weights.append(gate_weights)
        
        # Proyección final a vocabulario usando BitLinear
        logits = self.output_projection(x)
        
        return logits, all_gate_weights 