#!/usr/bin/env python3
"""
corr_plot.py
Plots the Spearman correlation between experimental DMS scores and zero-shot ESM3 predictions,
con ajustes de presentación para densos conjuntos de puntos.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# — Rutas —
dir_script   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_script, '..', '..'))

data_dir       = os.path.join(project_root, 'results', 'compute_zero_shot_results')
csv_filename   = 'A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv'
csv_path       = os.path.join(data_dir, csv_filename)
output_filename = 'zero_shot_correlation_spearman.png'
output_path     = os.path.join(data_dir, output_filename)

# — Carga de datos —
df = pd.read_csv(csv_path)
x  = df['DMS_score']
y  = df['ESM3_open_score']

# — Jitter horizontal para dispersar puntos con el mismo DMS_score —
jitter = np.random.normal(loc=0.0, scale=0.02, size=len(x))
x_j = x + jitter

# — Cálculo de Spearman —
rho, pval = spearmanr(x, y)

# — Gráfico —
plt.figure(figsize=(8, 6))
plt.scatter(x_j, y,
            s=12,
            alpha=0.4,
            edgecolors='none',
            marker='o')

# Línea de regresión Pearson solo como guía visual (opcional)
m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 100)
plt.plot(xx, m*xx + b, linestyle='--', linewidth=1, label='Guía visual (Pearson)')

# Anotaciones
plt.title(f'Zero-Shot Spearman Correlation (ρ = {rho:.2f})')
plt.xlabel('Experimental DMS_score')
plt.ylabel('ESM3_open_score (Predicted)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()

# Guardar
os.makedirs(data_dir, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f'✅ Gráfica de Spearman guardada en {output_path}')
