#!/usr/bin/env python
# coding: utf-8
"""
05_transfer_regress_normalized.py
-------------------------
Entrenamiento ligero de la cabeza de regresión sobre embeddings de ESM-3,
con normalización del target (–log10 Kd) a media 0 y σ=1, seguimiento de
Spearman, MSE y R², exportación de predicciones y generación de todos los
plots (incluyendo scatter de Spearman) en una sola pasada.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from modelo_esm3_regresion import ESM3Regressor

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────────────
logs_dir = Path("../logs")
logs_dir.mkdir(exist_ok=True)
logfile = logs_dir / f"esm3_training_norm_full_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler()]
)
log = logging.getLogger(__name__)
log.info("=== Iniciando full pipeline: fine-tuning + plots normalizados ===")

# ────────────────────────────────────────────────────────────────────────
#  Hiper-parámetros y rutas
# ────────────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV_PATH        = "../data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"
MODEL_SAVE_PATH = "../model/esm3_regressor_kd_normalized.pt"
SCALER_PATH     = "../model/kd_scaler_normalized.joblib"
EMBED_DIR       = Path("../results/embeddings_token_final")
RESULTS_DIR     = Path("../results/transfer_learning_regression_normalized")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_DIM     = 1536
USE_ATTENTION = True
LR            = 1e-4
EPOCHS        = 300
BATCH_SIZE    = 32

torch.manual_seed(42)

# ────────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    """Devuelve (emb_tensor, kd_scaled)."""
    def __init__(self, df, emb_dict):
        self.df = df.reset_index(drop=True)
        self.emb_dict = emb_dict
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_id = row["sequence"]
        kd_scaled = torch.tensor(row["kd_scaled"], dtype=torch.float32)
        emb = self.emb_dict[seq_id]         # [L, D]
        emb = emb.unsqueeze(0)              # [1, L, D], igual que el otro script
        return emb.to(DEVICE), kd_scaled.to(DEVICE)


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

# —————————————————————————————————————————————
# Aquí va la única normalización de kd
scaler = StandardScaler()
train_df["kd_scaled"] = scaler.fit_transform(train_df[["kd"]])
test_df["kd_scaled"]  = scaler.transform(test_df[["kd"]])
joblib.dump(scaler, SCALER_PATH)
log.info(f"Scaler normalizado guardado → {SCALER_PATH}")
# —————————————————————————————————————————————

# # ────────────────────────────────────────────────────────────────────────
# 1) Cargar embeddings precalculados
# ────────────────────────────────────────────────────────────────────────
cands = sorted(EMBED_DIR.glob("concatenated_*.pt"))
if not cands:
    raise FileNotFoundError(f"No hay 'concatenated_*.pt' en {EMBED_DIR}")
EMBED_PATH = str(cands[0])
log.info(f"Cargando embeddings de: {EMBED_PATH}")

EMBED_DICT = torch.load(EMBED_PATH, map_location="cpu")
log.info(f"→ {len(EMBED_DICT)} secuencias en cache")

# ────────────────────────────────────────────────────────────────────────
# 3) Dataset & DataLoader
# ────────────────────────────────────────────────────────────────────────
train_ds = ProteinDataset(train_df, EMBED_DICT)
test_ds  = ProteinDataset(test_df,  EMBED_DICT)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
# emb_batch tendrá forma [B, 1, L, D]

# ────────────────────────────────────────────────────────────────────────
#  3) Instanciar modelo
# ────────────────────────────────────────────────────────────────────────
regressor = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to(DEVICE)
optimizer = torch.optim.Adam(regressor.parameters(), lr=LR)
loss_fn   = nn.MSELoss()

# ────────────────────────────────────────────────────────────────────────
#  4) Entrenamiento + Evaluación
# ────────────────────────────────────────────────────────────────────────
train_losses, test_losses, r2_scores, spearman_scores = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    # ——— Train ——————————————————————————————————————————————————————
    regressor.train()
    running_loss = 0.0
    for emb_batch, kd_scaled in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
        # emb_batch: [B, 1, L, D], kd_scaled: [B]
        preds = regressor(emb_batch).squeeze(-1)  # [B]
        loss = loss_fn(preds, kd_scaled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * emb_batch.size(0)
    train_losses.append(running_loss / len(train_dl.dataset))

    # ——— Eval ——————————————————————————————————————————————————————
    regressor.eval()
    preds_orig, targets_orig = [], []
    with torch.no_grad():
        for emb_batch, kd_scaled in test_dl:
            p = regressor(emb_batch).squeeze(-1)
            p_np = p.cpu().numpy().reshape(-1,1)
            k_np = kd_scaled.cpu().numpy().reshape(-1,1)
            orig_p = scaler.inverse_transform(p_np).flatten()
            orig_k = scaler.inverse_transform(k_np).flatten()

            preds_orig.extend(orig_p.tolist())
            targets_orig.extend(orig_k.tolist())

    mse      = mean_squared_error(targets_orig, preds_orig)
    r2       = r2_score(targets_orig, preds_orig)
    spearman = spearmanr(targets_orig, preds_orig).correlation

    test_losses.append(mse)
    r2_scores.append(r2)
    spearman_scores.append(spearman)

    log.info(
        f"Epoch {epoch:3d} | "
        f"TrainLoss(scaled)={train_losses[-1]:.4f}  | "
        f"TestMSE(orig)={mse:.4f}  | R2={r2:.4f}  | Spearman={spearman:.4f}"
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
        spearman_scores=spearman_scores,
        hyperparameters=dict(
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LR,
            embed_dim=EMBED_DIM,
            use_attention=USE_ATTENTION
        )
    ),
    "../model/training_metrics_normalized.pt"
)
log.info(f"Modelo normalizado guardado en {MODEL_SAVE_PATH}")

# ────────────────────────────────────────────────────────────────────────
#  6) Exportar CSV de predicciones en test
# ────────────────────────────────────────────────────────────────────────
regressor.eval()
out_df = test_df.reset_index(drop=True).copy()
all_preds_scaled, all_kd = [], []
with torch.no_grad():
    for emb_batch, kd_scaled in test_dl:
        p_scaled = regressor(emb_batch).cpu().numpy()  # [B]
        all_preds_scaled.extend(p_scaled.tolist())
        all_kd.extend(kd_scaled.cpu().numpy().tolist())
# desescalar todo de golpe
preds_kd = scaler.inverse_transform(
    np.array(all_preds_scaled).reshape(-1,1)
).flatten()
out_df["pred_scaled"] = all_preds_scaled
out_df["pred_kd"]     = preds_kd

csv_path = RESULTS_DIR / "test_predictions_normalized.csv"
out_df.to_csv(csv_path, index=False)
log.info(f"Predicciones normalizadas guardadas en {csv_path}")

final_mse      = mean_squared_error(out_df["kd"], out_df["pred_kd"])
final_r2       = r2_score(out_df["kd"], out_df["pred_kd"])
final_spearman = spearmanr(out_df["kd"], out_df["pred_kd"]).correlation
log.info(f"Final Test — MSE={final_mse:.4f}, R2={final_r2:.4f}, Spearman={final_spearman:.4f}")

# ────────────────────────────────────────────────────────────────────────
#  7) Graficar todas las curvas y el scatter de Spearman
# ────────────────────────────────────────────────────────────────────────
epochs = list(range(1, EPOCHS + 1))

# — Pérdida & MSE
plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses,   label='Train Loss (scaled)')
plt.plot(epochs, test_losses,    label='Test MSE (orig)')
plt.xlabel("Época"); plt.ylabel("MSE")
plt.title("Curvas de aprendizaje normalizadas")
plt.legend()
plt.annotate(f"Final MSE={final_mse:.3f}", xy=(0.7, 0.9), xycoords='axes fraction')
plt.savefig(RESULTS_DIR / "curvas_aprendizaje_normalized.png", dpi=300, bbox_inches='tight')
plt.close()

# — R²
plt.figure(figsize=(10,6))
plt.plot(epochs, r2_scores, label='R² Score')
plt.xlabel("Época"); plt.ylabel("R²"); plt.ylim(0,1)
plt.title("Evolución del coeficiente R² normalizado")
plt.legend()
plt.annotate(f"Final R²={final_r2:.3f}", xy=(0.7, 0.1), xycoords='axes fraction')
plt.savefig(RESULTS_DIR / "r2_scores_normalized.png", dpi=300, bbox_inches='tight')
plt.close()

# — Spearman ρ
plt.figure(figsize=(10,6))
plt.plot(epochs, spearman_scores, label='Spearman ρ')
plt.xlabel("Época"); plt.ylabel("ρ"); plt.ylim(0,1)
plt.title("Evolución del coeficiente de correlación de rangos (Spearman) normalizado")
plt.legend()
plt.annotate(f"Final ρ={final_spearman:.3f}", xy=(0.7, 0.1), xycoords='axes fraction')
plt.savefig(RESULTS_DIR / "spearman_scores_normalized.png", dpi=300, bbox_inches='tight')
plt.close()

# — Scatter Spearman final
x = out_df["kd"]
y = out_df["pred_kd"]
jitter = np.random.normal(loc=0.0, scale=0.02, size=len(x))
x_j = x + jitter

plt.figure(figsize=(8,6))
plt.scatter(x_j, y, s=20, alpha=0.5, edgecolors='none')
m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 100)
plt.plot(xx, m*xx + b, linestyle='--', linewidth=1, label='Guía Pearson')
plt.title(f'Normalized Spearman Scatter (ρ = {final_spearman:.2f})')
plt.xlabel('Experimental –log10(Kd)')
plt.ylabel('Predicción –log10(Kd)')
plt.legend(); plt.grid(ls='--', lw=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "spearman_scatter_normalized.png", dpi=300)
plt.close()

log.info(f"Todas las salidas normalizadas se guardaron en {RESULTS_DIR}")
