import torch
import pandas as pd
import os
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM3.from_pretrained("esm3-open").to(device)
logits_config = LogitsConfig(
    sequence=True, #logits en modo secuencia
    return_embeddings=True,     # Devuelve los embeddings promediados por token
    return_hidden_states=False
)

# Ruta al archivo CSV de entrada con columnas: mutant, sequence, DMS_score
csv_path = "/media/nova/datos/diego/test/test_ad/250610_esm3/data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"
# Ruta de salida a los promedios de embeddings, reusltado comprimido en un tar
output_dir = "/media/nova/datos/diego/test/test_ad/250610_esm3/results/embeddings_avg"
os.makedirs(output_dir, exist_ok=True)

# Verifica que exista la columna 'sequence'
df = pd.read_csv(csv_path)
assert "sequence" in df.columns, "El archivo CSV debe tener una columna 'sequence'"
sequences = df["sequence"].tolist()

# Procesar en batches, itero de 1000 en 1000 para evitar problemas de memoria
batch_size = 1000 

#tqdm muestra barra de progreso en el bucle, avanza conforme se procesan los batches
for start in tqdm(range(0, len(sequences), batch_size)):
    chunk = sequences[start:start + batch_size]
    embedding_map = {}

    for seq in chunk:
        try:
            protein = ESMProtein(sequence=seq)
            encoded = model.encode(protein)
            logits = model.logits(encoded, logits_config) #obtengo logits/embeddings
            emb = logits.embeddings.squeeze(0)  # Para tener solo: [seq_len, hidden_size] (elimino primera dimensión de batch)
            mean_emb = emb.mean(dim=0)          # Calculo vector medio a lo largo de [hidden_size], es decir, promedio los embeddings de todas las posiciones de la secuencia
            #ES UNA REPRESENTACION GLOBAL, resumen de la proteina entera, util para clustering de secuencias o busqueda de proteinas similares
            embedding_map[seq] = mean_emb.cpu() #lo muevo a la cpu y almaceno en diccionario embedding_map, bajo la clave de la propia secuencia
        except Exception as e:
            print(f"Error en secuencia: {seq[:10]}... {e}")

    torch.save(embedding_map, os.path.join(output_dir, f"batch_{start // batch_size}.pt"))

print("✅ Embeddings extraídos y guardados.")
