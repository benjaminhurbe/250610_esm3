#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit · ESM-3 Local + Light-Attention Regressor
---------------------------------------------------
* Usa exactamente el mismo modelo ESM3-open local que usaste para entrenar
* Replica la misma función embed_sequence del entrenamiento
* Garantiza embeddings idénticos a los del entrenamiento
"""
import os
import joblib, torch, streamlit as st
from pathlib import Path
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from scripts.modelo_esm3_regresion import ESM3Regressor

# ──────────────────────────────────────────
# CONFIG (igual que en tu entrenamiento)
# ──────────────────────────────────────────
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBED_DIM     = 1536
REG_WEIGHTS   = Path("model/esm3_regressor_kd_normalized.pt")
SCALER_PATH   = Path("model/kd_scaler_normalized.joblib")
USE_ATTENTION = True

torch.manual_seed(42)

# ──────────────────────────────────────────
# 1) Carga del modelo ESM3-open local (EXACTAMENTE igual que tu entrenamiento)
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_esm3_local():
    """Carga el mismo modelo ESM3-open que usaste para entrenar"""
    st.info("🔄 Cargando modelo ESM3-open local...")
    
    # EXACTAMENTE igual que en tu script de entrenamiento
    esm3_model = ESM3.from_pretrained("esm3-open").to(DEVICE)
    esm3_model.eval()
    
    # EXACTAMENTE la misma configuración que usaste
    logits_config = LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=False
    )
    
    st.success("✅ Modelo ESM3-open cargado exitosamente")
    return esm3_model, logits_config

# ──────────────────────────────────────────
# 2) Carga del regressor + scaler
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_regressor():
    reg = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to(DEVICE)
    reg.load_state_dict(torch.load(REG_WEIGHTS, map_location=DEVICE))
    reg.eval()
    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    return reg, scaler

# ──────────────────────────────────────────
# 3) FUNCIÓN IDÉNTICA a tu entrenamiento
# ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed_sequence(sequence):
    """
    COPIA EXACTA de tu función embed_sequence del entrenamiento
    """
    esm3_model, logits_config = load_esm3_local()
    
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        inputs = esm3_model.encode(protein).to(DEVICE)
        output = esm3_model.logits(inputs, logits_config)
        output_sq = output.embeddings.squeeze(0)
        mean_output = output_sq.mean(dim=0)
        return mean_output.to(DEVICE)  # [seq_len, embed_dim=1536]

# ──────────────────────────────────────────
# 4) Funciones de predicción con diferentes formatos
# ──────────────────────────────────────────
def predict_training_format(sequence):
    """Usa el formato exacto del entrenamiento: [1, seq_len, embed_dim]"""
    emb = embed_sequence(sequence).unsqueeze(0).unsqueeze(1)  # [1, seq_len, embed_dim]
    
    regressor, scaler = load_regressor()
    with torch.no_grad():
        pred_scaled = regressor(emb).item()
        pred_kd = scaler.inverse_transform([[pred_scaled]])[0, 0] if scaler else pred_scaled
    
    return pred_scaled, pred_kd, emb.shape



# ──────────────────────────────────────────
# 5) Utilidades Kd
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
# 6) Interfaz principal
# ──────────────────────────────────────────
st.title("🧬 ESM3-open Local · Predicción de Kd")

with st.expander("ℹ️ Información", expanded=True):
    st.markdown("""
Esta app utiliza localmente el modelo *esm3-open* localmente para extraer los embeddings de una secuencia de antígeno

**Formatos de tensor probados:**
- **Entrenamiento**: `[1, seq_len, embed_dim]` - Como en tu loop de entrenamiento
- **Inferencia**: `[1, 1, seq_len, embed_dim]` - Como en script imperativo
""")

# Secuencias de ejemplo del dataset
example_seqs = {
    "Ejemplo corto": "MSTNPKPQRKTKRNTNKRPVQIAVHGSATLQKYQVSLRSRLLKPAG",
    "Ejemplo medio": "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGSTAYAQKFQGRVTITADKSTNTAYMELSSLRSEDTAVYYCARHGNYYYYYGMDVWGQGTTVTVSS",
    "Ejemplo largo": "MTMDKSELVQKAKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTERNEKKQQMGKEYREKIEAELQDICNDVLELLDKYLIPNATQPESKVFYLKMKGDYFRYLSEVASGDNKQTTVSNSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEILNSPEKACSLAKTAFDEAIAELDTLNEESYKDSTLIMQLLRDNLTLWTSENQGDEGDAGEGEN"
}

selected_example = st.selectbox("Secuencia de ejemplo:", list(example_seqs.keys()))
default_seq = example_seqs[selected_example]

seq_input = st.text_area(
    label="Secuencia de entrada:",
    value=default_seq,
    height=120,
    placeholder="Escribe o pega una secuencia de aminoácidos..."
)

if seq_input.strip():
    st.info(f"📏 Longitud de secuencia: {len(seq_input.strip())} aminoácidos")

# Botones de predicción
col1, col2 = st.columns(2)

with col1:
    if st.button("🎯 Predecir (Formato Entrenamiento)", type="primary"):
        if not seq_input.strip():
            st.warning("⚠️ Introduce una secuencia válida.")
        else:
            try:
                with st.spinner("Calculando embedding con ESM3-open local..."):
                    pred_scaled, pred_kd, tensor_shape = predict_training_format(seq_input.strip())
                
                kd_m, kd_nM = kd_from_logkd(pred_kd)
                
                st.success("✅ Predicción con formato de entrenamiento")
                st.code(f"Tensor shape: {tensor_shape}")
                
                met1, met2 = st.columns(2)
                with met1:
                    st.metric("Pred. normalizada", f"{pred_scaled:.4f}")
                    st.metric("–log₁₀(Kd)", f"{pred_kd:.4f}")
                with met2:
                    st.metric("Kd (M)", f"{kd_m:.3e}")
                    st.metric("Kd (nM)", f"{kd_nM:.1f}")
                
                st.markdown(f"**Interpretación:** {interpret_kd(kd_nM)}")
                
            except Exception as e:
                st.error(f"❌ Error en predicción: {e}")

# Información de debugging
with st.expander("🔧 Información técnica"):
    try:
        regressor, scaler = load_regressor()
        
        st.markdown("**Configuración del modelo:**")
        st.code(f"""
Embedding dimension: {EMBED_DIM}
Use attention: {USE_ATTENTION}
Device: {DEVICE}
Scaler available: {scaler is not None}
        """)
        
        if scaler:
            st.markdown("**Estadísticas del scaler:**")
            st.code(f"""
Media: {scaler.mean_[0]:.4f}
Escala: {scaler.scale_[0]:.4f}
Rango datos: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]
            """)
        
        # Información del modelo regressor
        total_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
        st.markdown(f"**Parámetros entrenables del regressor:** {total_params:,}")
        
    except Exception as e:
        st.error(f"Error cargando información del modelo: {e}")

st.divider()
st.caption("🔄 Usando ESM3-open local - embeddings idénticos al entrenamiento")

# Instrucciones de uso
with st.expander("📝 Notas de uso"):
    st.markdown("""
**¿Cuál formato usar?**
- **Formato Entrenamiento**: Si quieres replicar exactamente la predicción durante entrenamiento
- **Formato Inferencia**: Si quieres replicar el script imperativo de inferencia

**Si las predicciones siguen siendo constantes:**
1. Verifica que el archivo de pesos del modelo sea correcto
2. Comprueba que el scaler esté cargando correctamente
3. Considera que el modelo pudo haber colapsado durante entrenamiento

**Para debugging:** Usa diferentes secuencias de longitudes variadas para verificar que el modelo responde a cambios en la entrada.
""")