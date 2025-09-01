#!/home/nova/anaconda3/envs/esm3/bin/python

import sys
import os
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

# Autentication of huggingface_hub
# Add your token to the environment variable HF_TOKEN
# Run the login() line ONLY the first time on a new machine/environment
# After the model is cached and credentials are saved, you can keep it commented

# login(token=os.getenv("HF_TOKEN"))

if len(sys.argv) >= 2:
    csv_path = sys.argv[1]
else:
    # Default path if no argument provided
    csv_path = "../data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device} ({torch.cuda.get_device_name(device)})")

# Configuration (same as in the average embedding script)
model = ESM3.from_pretrained("esm3-open").to(device) # 1.4B parameters
logits_config = LogitsConfig(
    sequence=True,
    return_embeddings=True,
    return_hidden_states=False
)

# Read CSV and verify it has 'sequence' column, then convert sequence column to list
df = pd.read_csv(csv_path)
assert "sequence" in df.columns
sequences = df["sequence"].tolist()

output_dir  = "../results/embeddings_token"
os.makedirs(output_dir, exist_ok=True)

batch_size = 1000
total_sequences = len(sequences)
num_batches = (total_sequences + batch_size - 1) // batch_size  # Correct batch calculation

for batch_idx in tqdm(range(num_batches)):
    start = batch_idx * batch_size
    end = min(start + batch_size, total_sequences)  # Ensures not exceeding the limit
    chunk = sequences[start:end]
    token_map = {}

    for seq in chunk:
        try:
            protein = ESMProtein(sequence=seq)
            encoded = model.encode(protein).to(device)
            logits  = model.logits(encoded, logits_config) # get token-level embeddings again
            # logits.embeddings: [1, seq_len, hidden_size]
            emb = logits.embeddings.squeeze(0).cpu() # remove one dimension to get: [seq_len, hidden_size].
                    # Move to CPU directly without averaging all embeddings
            token_map[seq] = emb
        except Exception as e:
            print(f"⚠️ Error in sequence {seq[:10]}... {e}")

    fname = f"tokens_{os.path.basename(csv_path).replace('.csv','')}_batch{start//batch_size}.pt"
    torch.save(token_map, os.path.join(output_dir, fname))

print("✅ Token-level embeddings saved.")
