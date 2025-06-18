#!/home/nova/anaconda3/envs/esm3/bin/python
# ======================================================
# SCRIPT 1: modelo_esm3_regresion.py
# Define el modelo Light Attention + MLP para regresión
# ======================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# Esta capa toma un tensor de entrada de la forma [B, L, D], donde B es el tamaño del batch, L es la longitud de la secuencia y D es la dimensión del embedding.
# Calcula para cada posición su peso, su importancia relativa, y así actualiza los embeddings de cada posición. 
# Finalmente, suma los embeddings ponderados para obtener un único embedding de salida por secuencia.
class LightAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [B, L, D]
        # En este caso attn_weights es una capa lineal que se aplica a cada posición de la secuencia, retornando un vector de atención para cada token o posición.
        scores = self.attn_weights(x).squeeze(-1)       # Retorna [batch, seq_len]
        # La sofmax simplemente convierte los scores en probabilidades. 
        weights = torch.softmax(scores, dim=1)
        # A continuacion multiplico [B, L, D] * [B, L, 1], de modo que el embedding de cada posicion se multiplica por el peso de esa posición
        # Luego sumo todo sobre la dimension de tokens de modo que tengo un embedding final para cada secuencia.
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # Resultado: [B, D]
        return context

class ESM3Regressor(nn.Module):
    def __init__(self, input_dim, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = LightAttention(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), # Reduce [B, D] a [B, 256]
            nn.ReLU(), # Introduce no linealidad
            nn.Dropout(0.2), # Regulariza evitando sobreajuste
            nn.Linear(256, 1) # Colapsa en un solo valor por muestra, la predicción
        )
    # Batch, Length, Embed_dim    
    def forward(self, x):
        if self.use_attention:
            x = self.attn(x)
        else:
            x = x.mean(dim=1)
        # x: [B, D] después de la atención o el promedio
        return self.mlp(x).squeeze(-1)
