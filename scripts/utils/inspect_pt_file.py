import torch

"""
Este script inspecciona un archivo .pt que contiene embeddings de tokens extraídos de secuencias de proteínas.
"""

# Ruta al archivo .pt
pt_file = "/media/nova/datos/diego/test/250610_esm3/results/embeddings_avg/batch_0.pt"

# Cargar el contenido a un diccionario, que es data
data = torch.load(pt_file)

# Mostrar las claves (las secuencias)
print("Total de secuencias:", len(data))
print("\nPrimeras 5 secuencias y shape del embedding:")

for i, (seq, emb) in enumerate(data.items()):
    print(f"\nSecuencia {i+1}: {seq[:30]}...")  # Muestra primeros 30 caracteres
    print(f"Shape del embedding: {emb.shape}")  # [hidden_size] o [seq_len, hidden_size]
    if i >= 4:
        break
# Mostrar el tipo de datos de los embeddings
print("\nTipo de datos de los embeddings:", type(next(iter(data.values()))))
# Mostrar el dispositivo de los embeddings
print("Dispositivo de los embeddings:", next(iter(data.values())).device)
# Mostrar el tamaño del primer embedding
print("Tamaño del primer embedding:", next(iter(data.values())).size())
# Mostrar el tipo de datos del primer embedding
print("Tipo de datos del primer embedding:", next(iter(data.values())).dtype)
# Mostrar el número de tokens en el primer embedding (equivale al numero de aminoacidos)
print("Número de tokens en el primer embedding:", next(iter(data.values())).size(0) if next(iter(data.values())).dim() > 1 else 1)# Mostrar el primer embedding
print("Primer embedding (primeros 5 valores):", next(iter(data.values()))[:5])  # Muestra los primeros 5 valores del primer embedding
