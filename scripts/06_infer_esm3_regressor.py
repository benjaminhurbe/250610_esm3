#!/usr/bin/env python
# coding: utf-8
"""
infer_esm3_regressor.py

Inferencia con un modelo ESM3Regressor pre-entrenado (normalizado) y su scaler.
Predice –log10(Kd) para dos secuencias de ejemplo y guarda:
  - CSV con seq_id, kd_real, kd_predicho.
  - Scatter plot con línea identidad en rojo y anotaciones.
"""

import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from modelo_esm3_regresion import ESM3Regressor

# ────────────────────────────────────────────────────────────────────────
# Parámetros y rutas
# ────────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV_PATH     = Path("../data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv")
EMBED_DIR    = Path("../results/embeddings_token_final")
MODEL_PATH   = Path("../model/esm3_regressor_kd_normalized.pt")
SCALER_PATH  = Path("../model/kd_scaler_normalized.joblib")
OUT_DIR      = Path("../results/transfer_learning_regression_normalized")
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# 1) Carga de embeddings (dinámico)
# ────────────────────────────────────────────────────────────────────────
cands = sorted(EMBED_DIR.glob("concatenated_*.pt"))
if not cands:
    raise FileNotFoundError(f"No embeddings en {EMBED_DIR}")
EMBED_PATH = cands[0]
print(f"→ Cargando embeddings de: {EMBED_PATH.name}")
EMBED_DICT = torch.load(EMBED_PATH, map_location="cpu")  # {seq_id: Tensor[L,D]}

# ────────────────────────────────────────────────────────────────────────
# 2) Montar modelo y cargar pesos
# ────────────────────────────────────────────────────────────────────────
print("→ Cargando modelo…")
model = ESM3Regressor(input_dim=1536, use_attention=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ────────────────────────────────────────────────────────────────────────
# 3) Cargar scaler
# ────────────────────────────────────────────────────────────────────────
print("→ Cargando scaler…")
scaler = joblib.load(SCALER_PATH)

# ────────────────────────────────────────────────────────────────────────
# 4) Preparar datos de inferencia
# ────────────────────────────────────────────────────────────────────────
df = (
    pd.read_csv(CSV_PATH, usecols=["sequence","DMS_score"])
      .dropna()
      .rename(columns={"DMS_score":"kd"})
)

# Secuencia 1 = primera fila
seq1, kd1 = df.iloc[0]["sequence"], df.iloc[0]["kd"]
# Secuencia 2 = donde kd == 7.0
subset7 = df[df["kd"] == 7.0]
if not subset7.empty:
    seq2, kd2 = subset7.iloc[0]["sequence"], 7.0
else:
    seq2, kd2 = None, None

examples = [(seq1, kd1)]
if seq2 is not None:
    examples.append((seq2, kd2))

# ────────────────────────────────────────────────────────────────────────
# 5) Inferir y recopilar resultados
# ────────────────────────────────────────────────────────────────────────
results = []
for seq_id, true_kd in examples:
    # construir tensor [1,1,L,D]
    emb = EMBED_DICT[seq_id].unsqueeze(0).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(emb).squeeze().cpu().item()
    pred_kd = scaler.inverse_transform([[pred_scaled]])[0,0]
    print(f"→ {seq_id[:10]}…: real={true_kd:.3f}  pred={pred_kd:.3f}")
    results.append({"sequence": seq_id, "kd_real": true_kd, "kd_pred": pred_kd})

# ────────────────────────────────────────────────────────────────────────
# 6) Guardar CSV de inferencia
# ────────────────────────────────────────────────────────────────────────
out_csv = OUT_DIR / "inference_results.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"→ CSV de inferencia guardado en {out_csv}")

# ────────────────────────────────────────────────────────────────────────
# 7) Graficar scatter real vs predicho con línea identidad roja
# ────────────────────────────────────────────────────────────────────────
x = np.array([r["kd_real"] for r in results])
y = np.array([r["kd_pred"] for r in results])
j = np.random.normal(scale=0.01, size=len(x))  # un poco de jitter para visibilidad
xj = x + j

plt.figure(figsize=(6,6))
plt.scatter(xj, y, s=50, alpha=0.8, label="Puntos inferidos")
# guía de Pearson
m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 100)
plt.plot(xx, m*xx + b, "--", lw=1, label="Regresión lineal")
# línea identidad y=x en rojo
plt.plot(xx, xx, color="red", lw=1, label="Identidad (y=x)")
for xi, yi, seq_id in zip(xj, y, [r["sequence"] for r in results]):
    plt.annotate(seq_id[:6]+"…", (xi, yi), textcoords="offset points", xytext=(5,-5))
plt.xlabel("Experimental –log10(Kd)")
plt.ylabel("Predicho –log10(Kd)")
plt.title("Inferencia ESM-3 Regressor")
plt.legend(); plt.grid(ls="--", lw=0.5, alpha=0.6)
plt.tight_layout()

out_png = OUT_DIR / "inference_scatter.png"
plt.savefig(out_png, dpi=300)
print(f"→ Scatter guardado en {out_png}")
