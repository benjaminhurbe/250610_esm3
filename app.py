#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit · ESM-3 + Light-Attention Regressor
--------------------------------------------
* Introduce una secuencia → obtiene embedding con la API moderna de ESM-3
* Pasa por tu cabeza de regresión
* Devuelve: pred_scaled, –log10 Kd (desnormalizado si aplica), Kd (M y nM) e interpretación
"""
import os
import joblib, torch, streamlit as st
from pathlib import Path
import esm
from esm.sdk.api import ESMProtein, LogitsConfig
from scripts.modelo_esm3_regresion import ESM3Regressor   # mi clase
from huggingface_hub import login


# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBED_DIM     = 1536
REG_WEIGHTS   = Path("model/esm3_regressor_kd_normalized.pt")
SCALER_PATH   = Path("model/kd_scaler_normalized.joblib")    # whatevert path beacuse in this case there is no scaler
USE_ATTENTION = True

torch.manual_seed(42)


# ──────────────────────────────────────────
# 1) Carga perezosa de ESM-3
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_esm3():
    token = os.getenv("ESM_API_TOKEN") or st.secrets.get("esm")
    model = esm.sdk.client("esm3-small-2024-08", token=token)
    logits_cfg = LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=False
    )
    return model, logits_cfg

# ──────────────────────────────────────────
# 2) Carga del regressor + (opcional) scaler
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_regressor():
    reg = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to(DEVICE)
    reg.load_state_dict(torch.load(REG_WEIGHTS, map_location=DEVICE))
    reg.eval()
    # scaler es opcional: si no existe, retornamos None
    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    return reg, scaler

# ──────────────────────────────────────────
# 3) Embedding de una secuencia
# ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_embedding(seq: str):
    model, logits_cfg = load_esm3()
    seq = seq.replace(" ", "").upper()
    with torch.no_grad():
        protein = ESMProtein(sequence=seq)
        inp     = model.encode(protein).to(DEVICE)         # [1, L]  tokens
        out     = model.logits(inp, logits_cfg)
        emb_LD  = out.embeddings.squeeze(0)                 # [L, D=1536]
    return emb_LD.unsqueeze(0).float()                              # [1, L, D]

# ──────────────────────────────────────────
# 4) Utilidades Kd
# ──────────────────────────────────────────
def kd_from_logkd(log_kd):
    kd_m  = 10 ** (-log_kd)
    kd_nM = kd_m * 1e9
    return kd_m, kd_nM

def interpret_kd(kd_nM: float) -> str:
    if kd_nM < 1:
        return "Afinidad **ultra-alta** (< 1 nM)"
    elif kd_nM < 10:
        return "Afinidad **muy alta** (1-10 nM)"
    elif kd_nM < 100:
        return "Afinidad **alta** (10-100 nM)"
    elif kd_nM < 1000:
        return "Afinidad **moderada** (0.1-1 µM)"
    else:
        return "Afinidad **baja** (> 1 µM)"

# ──────────────────────────────────────────
# 5) Interfaz
# ──────────────────────────────────────────
st.title("🧬 Predicción de Kd con ESM-3")

with st.expander("ℹ️ Instrucciones", expanded=True):
    st.markdown("""
1. Escribe/pega una secuencia de aminoácidos (**A-Z**, sin números).  
2. Pulsa **_Predecir_**.  
3. La app calcula el embedding con ESM-3 y muestra –log10 Kd, Kd (M / nM) y una
   interpretación cualitativa de la afinidad.
""")

seq_input = st.text_area(
    label   = "Secuencia de entrada",
    value   = "",
    height  = 140,
    placeholder = "MSTNPKPQRKTK…"
)

if st.button("🔥 Predecir"):
    if not seq_input.strip():
        st.warning("⚠️ Introduce una secuencia válida.")
        st.stop()

    with st.spinner("Obteniendo embedding con ESM-3…"):
        emb = compute_embedding(seq_input.strip())

    regressor, scaler = load_regressor()
    with torch.no_grad():
        pred_scaled = regressor(emb.to(DEVICE)).item()
        # Des-normaliza si existe scaler
        log_kd = scaler.inverse_transform([[pred_scaled]])[0, 0] if scaler else pred_scaled

    kd_m, kd_nM = kd_from_logkd(log_kd)

    # ── Resultados ────────────────────────────────────────────────
    st.success("✅ Predicción completada")
    st.metric("Predicción (z-score)" if scaler else "Predicción –log₁₀(Kd)",
              f"{pred_scaled: .3f}")
    if scaler:
        st.metric("–log₁₀(Kd)", f"{log_kd: .3f}")

    c1, c2 = st.columns(2)
    c1.metric("Kd (M)",  f"{kd_m: .3e}")
    c2.metric("Kd (nM)", f"{kd_nM: .1f}")

    st.markdown(f"### Interpretación\n{interpret_kd(kd_nM)}")

st.divider()
st.caption("App construida con 🐍 PyTorch + 🌐 Streamlit  |  Embedding = 1536-D")
