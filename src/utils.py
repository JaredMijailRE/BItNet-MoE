import torch
import torch.nn as nn
import torch.optim as optim
from .Transformer import MoETransformer
from .LoadBalancing import calculate_load_balancing_loss

class MoETrainer:
    """Clase para entrenar el modelo MoE Transformer con cuantización BitLinear"""
    
    def __init__(self, model, optimizer, device='cuda', load_balance_weight=0.01):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.load_balance_weight = load_balance_weight
        self.model.to(device)
        
    def compute_loss(self, logits, targets, all_gate_weights):
        """Calcula la pérdida total incluyendo balanceado de carga"""
        # Pérdida principal de predicción de tokens
        main_loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # Pérdida auxiliar de balanceado de carga
        aux_loss = 0.0
        for gate_weights in all_gate_weights:
            # Aplanar gate_weights para el cálculo
            gate_weights_flat = gate_weights.view(-1, gate_weights.size(-1))
            aux_loss += calculate_load_balancing_loss(
                gate_weights_flat, 
                gate_weights.size(-1)
            )
        
        # Promedio de pérdida auxiliar entre capas
        aux_loss = aux_loss / len(all_gate_weights)
        
        # Pérdida total
        total_loss = main_loss + self.load_balance_weight * aux_loss
        
        return total_loss, main_loss, aux_loss
    
    def train_step(self, input_ids, targets, attention_mask=None):
        """Un paso de entrenamiento"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, all_gate_weights = self.model(input_ids, attention_mask)
        
        # Calcular pérdida
        total_loss, main_loss, aux_loss = self.compute_loss(
            logits, targets, all_gate_weights
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'aux_loss': aux_loss.item()
        }
    
    def eval_step(self, input_ids, targets, attention_mask=None):
        """Un paso de evaluación"""
        self.model.eval()
        
        with torch.no_grad():
            logits, all_gate_weights = self.model(input_ids, attention_mask)
            total_loss, main_loss, aux_loss = self.compute_loss(
                logits, targets, all_gate_weights
            )
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'aux_loss': aux_loss.item()
        }


def create_moe_model(vocab_size, embed_dim=512, num_heads=8, num_layers=6, 
                     num_experts=8, k=2, max_seq_len=512, dropout=0.1):
    """Función helper para crear un modelo MoE Transformer con BitLinear"""
    return MoETransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        k=k,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

def create_trainer(model, learning_rate=1e-4, weight_decay=0.01, 
                   device='cuda', load_balance_weight=0.01):
    """Función helper para crear un trainer para modelos MoE con BitLinear"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return MoETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        load_balance_weight=load_balance_weight
    ) 