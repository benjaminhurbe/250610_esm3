import os
import torch
from collections import defaultdict

# Configuración
#EMBEDDINGS_DIR = "/media/nova/datos/proj_esm3_proof/250610_esm3/results/embeddings_token"
EMBEDDINGS_DIR = "../../results/embeddings_avg"
CONCATENATED_FILE = "concatenated_embeddings_final.pt"

# Contadores
total_sequences = 0
sequences_per_prefix = defaultdict(int)
file_counts = []

print(f"Analizando directorio: {EMBEDDINGS_DIR}\n")

# Iterar sobre todos los archivos .pt
for filename in os.listdir(EMBEDDINGS_DIR):
    if filename.endswith(".pt") and filename != CONCATENATED_FILE:
        filepath = os.path.join(EMBEDDINGS_DIR, filename)
        
        try:
            # Cargar el archivo (solo para contar, sin almacenar datos)
            data = torch.load(filepath, map_location='cpu')
            
            # Determinar tipo de datos
            if isinstance(data, dict):
                count = len(data)
            elif isinstance(data, list):
                count = len(data)
            elif isinstance(data, torch.Tensor):
                count = data.size(0)
            else:
                count = 1  # Para tipos desconocidos
            
            # Actualizar contadores
            total_sequences += count
            prefix = filename.split("_")[2]  # Ej: tokens_part_aa_batch0.pt -> aa
            sequences_per_prefix[prefix] += count
            file_counts.append((filename, count))
            
            print(f"{filename}: {count} secuencias")
            
        except Exception as e:
            print(f"⚠️ Error al procesar {filename}: {str(e)}")
            continue

# Resultados
print("\n" + "="*50)
print(f"TOTAL SECUENCIAS (excluyendo concatenado): {total_sequences}")
print("="*50 + "\n")

# Resumen por prefijo
for prefix, count in sequences_per_prefix.items():
    print(f"Prefijo '{prefix}': {count} secuencias")

# Archivos con más secuencias (top 5)
print("\nArchivos con más secuencias:")
for filename, count in sorted(file_counts, key=lambda x: x[1], reverse=True)[:5]:
    print(f"- {filename}: {count} secuencias")
    
# Archivos con menos secuencias (top 5)
print("\nArchivos con menos secuencias:")
for filename, count in sorted(file_counts, key=lambda x: x[1])[:5]:
    print(f"- {filename}: {count} secuencias")