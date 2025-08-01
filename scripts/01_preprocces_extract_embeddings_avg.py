import torch
import pandas as pd
import os
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM3.from_pretrained("esm3-open").to(device)
logits_config = LogitsConfig(
    sequence=True,              # logits in sequence mode
    return_embeddings=True,     # Returns token-level embeddings
    return_hidden_states=False
)

# PATH of the csv input file with columns: mutant, sequence, DMS_score
csv_path = "/media/nova/datos/diego/test/test_ad/250610_esm3/data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"
# Output path (results in .tar)
output_dir = "/media/nova/datos/diego/test/test_ad/250610_esm3/results/embeddings_avg"
os.makedirs(output_dir, exist_ok=True)

# Verification of sequence column
df = pd.read_csv(csv_path)
assert "sequence" in df.columns, "El archivo CSV debe tener una columna 'sequence'"
sequences = df["sequence"].tolist()

# Batch processing
batch_size = 1000 

#tqdm use for progress bar display
for start in tqdm(range(0, len(sequences), batch_size)):
    chunk = sequences[start:start + batch_size]
    embedding_map = {}

    for seq in chunk:
        try:
            protein = ESMProtein(sequence=seq)
            encoded = model.encode(protein)
            logits = model.logits(encoded, logits_config) #obtaining of logits/embeddings
            emb = logits.embeddings.squeeze(0)  # Squeezing to obtain the structure [seq_len, hidden_size]
            mean_emb = emb.mean(dim=0)          # Computing mean vector across sequence length, averaging all positions to get global representation
                                                # Useful for clustering or searching for similar proteins.
            embedding_map[seq] = mean_emb.cpu() # Move to CPU and store in the embedding_map dictionary, using the sequence itself as the key.
        except Exception as e:
            print(f"Error en secuencia: {seq[:10]}... {e}")

    torch.save(embedding_map, os.path.join(output_dir, f"batch_{start // batch_size}.pt"))

print("✅ Embeddings extraídos y guardados.")
