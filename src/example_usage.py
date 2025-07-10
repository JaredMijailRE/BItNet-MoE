"""
Ejemplo de uso del MoE Transformer con cuantización BitLinear
"""

import torch
from src import create_moe_model, create_trainer

def ejemplo_entrenamiento():
    """Ejemplo completo de cómo entrenar el modelo MoE con BitLinear"""
    
    # Configuración del modelo
    vocab_size = 50000  # Tamaño del vocabulario
    embed_dim = 512     # Dimensión de embeddings
    num_heads = 8       # Número de cabezas de atención
    num_layers = 6      # Número de capas transformer
    num_experts = 8     # Número de expertos en cada capa MoE
    k = 2              # Número de expertos activos por token
    max_seq_len = 512  # Longitud máxima de secuencia
    
    # Crear modelo MoE con BitLinear
    print("Creando modelo MoE Transformer con BitLinear...")
    model = create_moe_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        k=k,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # Mostrar información del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Crear trainer
    print("\nCreando trainer...")
    trainer = create_trainer(
        model=model,
        learning_rate=1e-4,
        weight_decay=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        load_balance_weight=0.01  # Peso para pérdida de balanceado
    )
    
    # Datos de ejemplo (simulados)
    batch_size = 4
    seq_len = 128
    device = trainer.device
    
    print(f"\nUsando dispositivo: {device}")
    
    # Generar datos de ejemplo
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"Forma de entrada: {input_ids.shape}")
    print(f"Forma de objetivos: {targets.shape}")
    
    # Paso de entrenamiento
    print("\nEjecutando paso de entrenamiento...")
    loss_info = trainer.train_step(input_ids, targets)
    
    print("Resultados del entrenamiento:")
    print(f"  Pérdida total: {loss_info['total_loss']:.4f}")
    print(f"  Pérdida principal: {loss_info['main_loss']:.4f}")
    print(f"  Pérdida auxiliar (balanceado): {loss_info['aux_loss']:.4f}")
    
    # Paso de evaluación
    print("\nEjecutando paso de evaluación...")
    eval_info = trainer.eval_step(input_ids, targets)
    
    print("Resultados de la evaluación:")
    print(f"  Pérdida total: {eval_info['total_loss']:.4f}")
    print(f"  Pérdida principal: {eval_info['main_loss']:.4f}")
    print(f"  Pérdida auxiliar (balanceado): {eval_info['aux_loss']:.4f}")
    
    print("\n✅ Ejemplo completado exitosamente!")
    
    return model, trainer

def ejemplo_inferencia():
    """Ejemplo de inferencia con el modelo"""
    
    # Crear modelo pequeño para inferencia
    model = create_moe_model(
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_experts=4,
        k=1
    )
    
    model.eval()
    
    # Datos de entrada
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"Entrada: {input_ids.shape}")
    
    # Inferencia
    with torch.no_grad():
        logits, gate_weights = model(input_ids)
    
    print(f"Salida (logits): {logits.shape}")
    print(f"Número de capas con gate weights: {len(gate_weights)}")
    print(f"Forma de gate weights por capa: {gate_weights[0].shape}")
    
    # Generar predicciones
    predictions = torch.argmax(logits, dim=-1)
    print(f"Predicciones: {predictions.shape}")
    
    return model, logits, gate_weights

if __name__ == "__main__":
    print("🚀 Ejemplo de MoE Transformer con BitLinear")
    print("=" * 50)
    
    # Ejecutar ejemplo de entrenamiento
    try:
        ejemplo_entrenamiento()
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
    
    print("\n" + "=" * 50)
    
    # Ejecutar ejemplo de inferencia
    try:
        print("🔮 Ejemplo de inferencia")
        ejemplo_inferencia()
    except Exception as e:
        print(f"Error en inferencia: {e}") 