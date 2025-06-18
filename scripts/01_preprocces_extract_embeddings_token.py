#!/home/nova/anaconda3/envs/esm3/bin/python

import sys
import os
import torch
import pandas as pd
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

assert len(sys.argv) >= 2, "Usage: python extract_token_embeddings_split.py <csv_file>"
#RUTA ES: /media/nova/datos/proj_esm3_proof/250610_esm3/data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv
csv_path = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Usando dispositivo: {device} ({torch.cuda.get_device_name(device)})")

# Configuración (igual que en el script de average embedding)
model = ESM3.from_pretrained("esm3-open").to(device) #1.4 B de parametros
logits_config = LogitsConfig(
    sequence=True,
    return_embeddings=True,
    return_hidden_states=False
)

#Leo csv y verfico que tenga la columna 'sequence', luego paso la columna de secuencias a una lista
df = pd.read_csv(csv_path)
assert "sequence" in df.columns
sequences = df["sequence"].tolist()

output_dir  = "../results/embeddings_token"
os.makedirs(output_dir, exist_ok=True)

batch_size = 1000
total_sequences = len(sequences)
num_batches = (total_sequences + batch_size - 1) // batch_size  # Cálculo correcto de batches

for batch_idx in tqdm(range(num_batches)):
    start = batch_idx * batch_size
    end = min(start + batch_size, total_sequences)  # Asegura no exceder el límite
    chunk = sequences[start:end]
    token_map = {}

    for seq in chunk:
        try:
            protein = ESMProtein(sequence=seq)
            encoded = model.encode(protein).to(device)
            logits  = model.logits(encoded, logits_config) #obtengo de nuevo embeddings por token
            # logits.embeddings: [1, seq_len, hidden_size]
            emb = logits.embeddings.squeeze(0).cpu() # elimino una dimension y me quedo con: [seq_len, hidden_size].
                    #Lo paso a cpu directamente sin sacar la media entre todos los embeddings
            token_map[seq] = emb
        except Exception as e:
            print(f"⚠️ Error en secuencia {seq[:10]}... {e}")

    fname = f"tokens_{os.path.basename(csv_path).replace('.csv','')}_batch{start//batch_size}.pt"
    torch.save(token_map, os.path.join(output_dir, fname))

print("✅ Embeddings token-level guardados.")
