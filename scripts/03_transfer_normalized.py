#!/usr/bin/env python
# coding: utf-8
"""
03_transfer_norm.py
-------------------
Fine-tuning ligero de una cabeza de regresión sobre embeddings de ESM-3,
con normalización del target (–log10 Kd) a media 0 y σ=1.
"""

import os, time, logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from modelo_esm3_regresion import ESM3Regressor

import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────────────
logs_dir = Path("../logs")
logs_dir.mkdir(exist_ok=True)
logfile = logs_dir / f"esm3_training_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler()]
)
log = logging.getLogger(__name__)
log.info("=== Iniciando fine-tuning ESM-3 (regresión Kd, target normalizado) ===")

# ────────────────────────────────────────────────────────────────────────
#  Hiper-parámetros y rutas
# ────────────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV_PATH        = "../data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"
MODEL_SAVE_PATH = "../model/esm3_regressor_kd.pt"
SCALER_PATH     = "../model/kd_scaler.joblib"
EMBED_DIR       = Path("../results/embeddings_token_final")

EMBED_DIM     = 1536
USE_ATTENTION = True
LR            = 1e-4
EPOCHS        = 200
BATCH_SIZE    = 32

torch.manual_seed(42)

# ────────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    """Devuelve (sequence_id, kd_scaled)."""
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["sequence"], torch.tensor(row["kd_scaled"], dtype=torch.float32)

# ────────────────────────────────────────────────────────────────────────
#  1) Cargar y normalizar Kd
# ────────────────────────────────────────────────────────────────────────
df = (
    pd.read_csv(CSV_PATH, usecols=["sequence", "DMS_score"])
      .dropna()
      .rename(columns={"DMS_score": "kd"})
)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
log.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

scaler = StandardScaler()
train_df["kd_scaled"] = scaler.fit_transform(train_df[["kd"]])
test_df ["kd_scaled"] = scaler.transform(   test_df [["kd"]])

joblib.dump(scaler, SCALER_PATH)
log.info(f"Scaler guardado → {SCALER_PATH}")

train_dl = DataLoader(ProteinDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(ProteinDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)

# ────────────────────────────────────────────────────────────────────────
#  2) Cargar embeddings precalculados
# ────────────────────────────────────────────────────────────────────────
cands = sorted(EMBED_DIR.glob("concatenated_*.pt"))
if not cands:
    raise FileNotFoundError(f"No hay 'concatenated_*.pt' en {EMBED_DIR}")
EMBED_PATH = str(cands[0])
log.info(f"Cargando embeddings de: {EMBED_PATH}")

EMBED_DICT = torch.load(EMBED_PATH, map_location="cpu")
log.info(f"→ {len(EMBED_DICT)} secuencias en cache")

def embed_batch_from_cache(seq_ids):
    """
    Devuelve tensor [B, 1, D]. Inserta seq_len=1 para la capa de atención.
    """
    embs = torch.stack([EMBED_DICT[s] for s in seq_ids])  # [B, D]
    return embs.unsqueeze(1).to(DEVICE)                   # [B, 1, D]

# ────────────────────────────────────────────────────────────────────────
#  3) Instanciar modelo
# ────────────────────────────────────────────────────────────────────────
regressor = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to(DEVICE)
optimizer = torch.optim.Adam(regressor.parameters(), lr=LR)
loss_fn   = nn.MSELoss()

# ────────────────────────────────────────────────────────────────────────
#  4) Entrenamiento + Evaluación
# ────────────────────────────────────────────────────────────────────────
train_losses, test_losses, r2_scores = [], [], []

for epoch in range(1, EPOCHS + 1):
    # ——— Train ——————————————————————————————————————————————————————
    regressor.train()
    epoch_loss = 0.0
    for seq_ids, kd_scaled in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
        kd_scaled = kd_scaled.to(DEVICE)
        embeds    = embed_batch_from_cache(list(seq_ids))
        preds     = regressor(embeds).squeeze(-1)

        loss = loss_fn(preds, kd_scaled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_dl))

    # ——— Eval ——————————————————————————————————————————————————————
    regressor.eval()
    preds_orig, targets_orig = [], []
    with torch.no_grad():
        for seq_ids, kd_scaled in test_dl:
            kd_s = kd_scaled.to(DEVICE)
            embs = embed_batch_from_cache(list(seq_ids))
            p    = regressor(embs).squeeze(-1)

            # — des-escalar a –log10(Kd)
            p_np = p.cpu().numpy().reshape(-1, 1)
            k_np = kd_s.cpu().numpy().reshape(-1, 1)
            orig_p = scaler.inverse_transform(p_np).flatten()
            orig_k = scaler.inverse_transform(k_np).flatten()

            preds_orig.extend(orig_p.tolist())
            targets_orig.extend(orig_k.tolist())

    mse = mean_squared_error(targets_orig, preds_orig)
    r2  = r2_score(targets_orig, preds_orig)
    test_losses.append(mse)
    r2_scores.append(r2)

    log.info(
        f"Epoch {epoch:3d} | "
        f"TrainLoss(scaled)={train_losses[-1]:.4f}  | "
        f"TestMSE(orig)={mse:.4f}  | R2={r2:.4f}"
    )

# ────────────────────────────────────────────────────────────────────────
#  5) Guardar modelo y métricas
# ────────────────────────────────────────────────────────────────────────
torch.save(regressor.state_dict(), MODEL_SAVE_PATH)
torch.save(
    dict(
        train_losses=train_losses,
        test_losses=test_losses,
        r2_scores=r2_scores,
        hyperparameters=dict(
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LR,
            embed_dim=EMBED_DIM,
            use_attention=USE_ATTENTION
        )
    ),
    "../model/training_metrics.pt"
)
log.info(f"Modelo guardado en {MODEL_SAVE_PATH}")

# ────────────────────────────────────────────────────────────────────────
#  6) Graficar curvas de aprendizaje y guardar dos plots
# ────────────────────────────────────────────────────────────────────────
plots_dir = Path("../results/transfer_learning_regression_head")
plots_dir.mkdir(parents=True, exist_ok=True)

epochs = list(range(1, EPOCHS + 1))

# — Curvas de pérdida
plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, 'b-', label='Train Loss (escalado)')
plt.plot(epochs, test_losses,  'r-', label='Test MSE (orig)')
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()
plt.title("Curvas de aprendizaje")
plt.savefig(plots_dir / "curvas_aprendizaje.png", dpi=300, bbox_inches='tight')
plt.close()
log.info(f"Gráfica de pérdidas guardada en {plots_dir}/curvas_aprendizaje.png")

# — Curva de R²
plt.figure(figsize=(10,6))
plt.plot(epochs, r2_scores, 'g-', label='R² Score')
plt.xlabel("Época")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.legend()
plt.title("Evolución del coeficiente R²")
plt.savefig(plots_dir / "r2_scores.png", dpi=300, bbox_inches='tight')
plt.close()
log.info(f"Gráfica de R² guardada en {plots_dir}/r2_scores.png")
