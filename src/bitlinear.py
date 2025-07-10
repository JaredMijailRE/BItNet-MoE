# créditos: https://github.com/kyegomez/BitNet/blob/main/kernel/gemm_lowbit.cpp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SimpleRMSNorm(nn.Module):
    """Normalización RMS simple para BitLinear"""
    
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calcular norma RMS
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return self.scale * x / (rms + self.eps)


def activation_quant(x: Tensor):
    """Cuantización por token a 8 bits. No se necesita agrupación.

    Args:
        x (Tensor): Tensor de activación a cuantizar

    Returns:
        Tensor: Tensor cuantizado
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor):
    """Cuantización de pesos binaria"""
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Linear):
    """
    Capa lineal personalizada con cuantización de bits.

    Args:
        in_features (int): Dimensión de entrada
        out_features (int): Dimensión de salida
        bias (bool): Si incluir sesgo o no
        **kwargs: Argumentos adicionales
    """

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        # Crear normalización RMS una sola vez
        self.norm = SimpleRMSNorm(in_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass de la capa BitLinear.

        Args:
            x (Tensor): Tensor de entrada

        Returns:
            Tensor: Tensor de salida cuantizado
        """
        w = self.weight
        
        # Normalizar entrada
        x_norm = self.norm(x)

        # STE (Straight Through Estimator) usando detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        # Operación lineal con tensores cuantizados
        y = F.linear(x_quant, w_quant, self.bias)
        return y


