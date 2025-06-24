#!/usr/bin/env python
# coding: utf-8

"""
train_regressor_without_norm_glob_with_plots.py
-----------------------------------------------
Fine-tuning del modelo ESM3 con cabeza de regresión,
sin normalizar el target (–log10 Kd).  
Carga automáticamente el primer “concatenated_*.pt” que encuentre en
../results/embeddings_token_final/, y al final exporta CSV y guarda
todas las figuras de métricas.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from modelo_esm3_regresion import ESM3Regressor

import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────
log_filename = f"esm3_training_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logging.info(msg)
    print(*args, **kwargs)

# ────────────────────────────────────────────────────────────────────────
# Hiper-parámetros y rutas
# ────────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV_PATH      = "../data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"
MODEL_SAVE    = "../model/esm3_regressor_kd.pt"
RESULTS_DIR   = Path("../results/transfer_learning_regression_head")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE    = 32
EPOCHS        = 300
LR            = 1e-4
USE_ATTENTION = True
EMBED_DIM     = 1536

log_print("=== Iniciando fine-tuning sin normalizar ===")
log_print(f"Usando dispositivo: {DEVICE}")

# ────────────────────────────────────────────────────────────────────────
# 1) Encontrar y cargar embeddings pre-computados
# ────────────────────────────────────────────────────────────────────────
EMBED_DIR = Path("../results/embeddings_token_final")
cands = sorted(EMBED_DIR.glob("concatenated_*.pt"))
if not cands:
    raise FileNotFoundError(f"No hay archivos 'concatenated_*.pt' en {EMBED_DIR}")
EMBED_PATH = str(cands[0])
log_print(f"🗂 Cargando embeddings desde {EMBED_PATH}…")
emb_dict = torch.load(EMBED_PATH, map_location="cpu")
log_print(f"✓ {len(emb_dict)} secuencias cargadas en cache")

# ────────────────────────────────────────────────────────────────────────
# 2) Preparar DataLoaders
# ────────────────────────────────────────────────────────────────────────
df = (
    pd.read_csv(CSV_PATH, usecols=["sequence","DMS_score"])
      .dropna()
      .rename(columns={"DMS_score":"kd"})
)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
log_print(f"▶️  Train: {len(train_df)} | Test: {len(test_df)} secuencias")

class RawDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        seq = self.df.loc[idx, "sequence"]
        kd  = self.df.loc[idx, "kd"]
        emb = emb_dict[seq]               # [seq_len, EMBED_DIM]
        return emb.unsqueeze(0).to(DEVICE), torch.tensor(kd, dtype=torch.float32).to(DEVICE)

train_dl = DataLoader(RawDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(RawDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)

# ────────────────────────────────────────────────────────────────────────
# 3) Instanciar modelo (solo la cabeza de regresión)
# ────────────────────────────────────────────────────────────────────────
regressor = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to(DEVICE)
optimizer = torch.optim.Adam(regressor.parameters(), lr=LR)
loss_fn   = nn.MSELoss()

# ────────────────────────────────────────────────────────────────────────
# 4) Entrenamiento + evaluación
# ────────────────────────────────────────────────────────────────────────
train_losses, test_losses = [], []
r2_scores, spearman_scores = [], []

for epoch in range(1, EPOCHS + 1):
    # — Train —
    regressor.train()
    running_loss = 0.0
    for emb_batch, kd_batch in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
        pred = regressor(emb_batch).squeeze(-1)
        loss = loss_fn(pred, kd_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * emb_batch.size(0)
    train_losses.append(running_loss / len(train_dl.dataset))

    # — Eval —
    regressor.eval()
    preds, targets = [], []
    with torch.no_grad():
        for emb_batch, kd_batch in tqdm(test_dl, desc=f"Epoch {epoch} [Eval]"):
            p = regressor(emb_batch).squeeze(-1).cpu().tolist()
            t = kd_batch.cpu().tolist()
            preds.extend(p); targets.extend(t)

    mse = mean_squared_error(targets, preds)
    r2  = r2_score(targets, preds)
    rho = spearmanr(targets, preds).correlation

    test_losses.append(mse)
    r2_scores.append(r2)
    spearman_scores.append(rho)

    print(
        f"Epoch {epoch:2d} — "
        f"TrainLoss={train_losses[-1]:.4f} | "
        f"TestMSE={mse:.4f} | R²={r2:.4f} | Spearman={rho:.4f}"
    )

# ────────────────────────────────────────────────────────────────────────
# 5) Guardar modelo entrenado
# ────────────────────────────────────────────────────────────────────────
torch.save(regressor.state_dict(), MODEL_SAVE)
log_print(f"✅ Modelo guardado en {MODEL_SAVE}")

# ────────────────────────────────────────────────────────────────────────
# 6) Exportar CSV de predicciones en test
# ────────────────────────────────────────────────────────────────────────
regressor.eval()
out_df = test_df.reset_index(drop=True).copy()
preds = []
with torch.no_grad():
    for emb_batch, _ in tqdm(DataLoader(RawDataset(test_df), batch_size=1), desc="Export preds"):
        p = regressor(emb_batch.to(DEVICE)).squeeze(-1).item()
        preds.append(p)

out_df["pred_kd"] = preds
csv_path = RESULTS_DIR / "test_predictions.csv"
out_df.to_csv(csv_path, index=False)
log_print(f"✅ Predicciones guardadas en {csv_path}")

final_mse      = mean_squared_error(out_df["kd"], out_df["pred_kd"])
final_r2       = r2_score(out_df["kd"], out_df["pred_kd"])
final_spearman = spearmanr(out_df["kd"], out_df["pred_kd"]).correlation
log_print(f"Final Test — MSE={final_mse:.4f}, R²={final_r2:.4f}, Spearman={final_spearman:.4f}")

# ────────────────────────────────────────────────────────────────────────
# 7) Graficar curvas y scatter de Spearman
# ────────────────────────────────────────────────────────────────────────
epochs = list(range(1, EPOCHS+1))

# — Curvas de pérdida & MSE
plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses,  label="Test MSE")
plt.xlabel("Época"); plt.ylabel("MSE")
plt.title("Curvas de aprendizaje (sin normalizar)")
plt.legend()
plt.annotate(f"Final MSE={final_mse:.3f}", xy=(0.7,0.9), xycoords="axes fraction")
plt.savefig(RESULTS_DIR/"curvas_aprendizaje.png", dpi=300, bbox_inches="tight")
plt.close()

# — R² Score
plt.figure(figsize=(10,6))
plt.plot(epochs, r2_scores, label="R² Score")
plt.xlabel("Época"); plt.ylabel("R²"); plt.ylim(0,1)
plt.title("Evolución del coeficiente R² (sin normalizar)")
plt.legend()
plt.annotate(f"Final R²={final_r2:.3f}", xy=(0.7,0.1), xycoords="axes fraction")
plt.savefig(RESULTS_DIR/"r2_scores.png", dpi=300, bbox_inches="tight")
plt.close()

# — Spearman ρ
plt.figure(figsize=(10,6))
plt.plot(epochs, spearman_scores, label="Spearman ρ")
plt.xlabel("Época"); plt.ylabel("ρ"); plt.ylim(0,1)
plt.title("Evolución del coeficiente de correlación de rangos (Spearman)")
plt.legend()
plt.annotate(f"Final ρ={final_spearman:.3f}", xy=(0.7,0.1), xycoords="axes fraction")
plt.savefig(RESULTS_DIR/"spearman_scores.png", dpi=300, bbox_inches="tight")
plt.close()

# — Scatter Spearman final
x = out_df["kd"].values
y = out_df["pred_kd"].values
jitter = np.random.normal(scale=0.02, size=len(x))
x_j = x + jitter

plt.figure(figsize=(8,6))
plt.scatter(x_j, y, s=20, alpha=0.5, edgecolors="none")
m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 100)
plt.plot(xx, m*xx + b, "--", linewidth=1, label="Guía Pearson")
plt.title(f"Spearman Scatter (ρ = {final_spearman:.2f})")
plt.xlabel("Experimental –log10(Kd)")
plt.ylabel("Predicción –log10(Kd)")
plt.legend(); plt.grid(ls="--", lw=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"spearman_scatter.png", dpi=300)
plt.close()

log_print(f"✅ Todas las figuras guardadas en {RESULTS_DIR}")
