import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_load_balancing_loss(gate_weights, num_experts):
    """Calcula la pérdida de balanceado de carga para evitar que algunos expertos
    sean subutilizados.

    Args:
        gate_weights: Tensor de forma [batch_size * seq_len, num_experts]
        num_experts: Número total de expertos.

    Returns:
        Tensor escalar de pérdida.
    """
    # gate_weights es la salida de la red de puerta (antes de top-k)
    # Necesitamos la probabilidad promedio de enrutamiento por experto
    # y la fracción de tokens enrutados a cada experto
    
    num_tokens = gate_weights.shape[0]
    
    # Calcular fracción de tokens enrutados a cada experto (f_i)
    # Usar los pesos directamente como proxy para el conteo de asignaciones
    # (Suma de pesos para cada experto a través de todos los tokens)
    tokens_per_expert = torch.sum(gate_weights, dim=0)  # Shape [num_experts]
    f_i = tokens_per_expert / num_tokens
    
    # Calcular probabilidad promedio de enrutamiento por experto (P_i)
    # Esta es la media de los pesos de puerta para cada experto
    mean_prob_per_expert = torch.mean(gate_weights, dim=0)  # Shape [num_experts]
    P_i = mean_prob_per_expert
    
    # Calcular la pérdida: alpha * num_experts * sum(f_i * P_i)
    # alpha es un hiperparámetro para escalar la pérdida
    loss = num_experts * torch.sum(f_i * P_i)
    return loss