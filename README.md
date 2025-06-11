# 🧬 ESM3: Pipeline de procesamiento de embeddings de proteínas

Este proyecto permite extraer embeddings y aplicar un análisis zero-shot a un dataset de 65 000 secuencias mutantes del antígeno CR9114 junto a un valor experimental de Kd de unión a Hemaglutinina subtipo H1. Los embeddings son extraidos a partir de las secuencias de aminoácidos usando el modelo [`esm3-open`](https://huggingface.co/EvolutionaryScale/esm3-open) de Evolutionary Scale, ejecutado localmente con GPU (si está disponible).

## 📁 Estructura del proyecto

250610_esm3/
├── data/
│   └── A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv  # Archivo con las secuencias
├── DMS_ids/
│   └── múltiples bases de datos de proteingym (*.csv)      # Bases de datos de DMS
├── reference_files/
│   └── clinical_substitutions.csv                          # csv's referencia de ProteinGym
│   └── clinical_indels.csv                                 # contienen la proteina target
│   └── DMS_indels.csv                                      # de ensayos DMS de indels y sustituciones
│   └── DMS_substitutions.csv                               # en esta base de datos se integró la de CR9114
├── scripts/
│   └── 01_preprocces_embeddings.py                         # Script de extracción de embeddings
├── embeddings_avg/                                         # Carpeta de salida con archivos .pt

## ⚙️ Requisitos

- Python 3.11 (recomendado con Conda)
- PyTorch (con soporte para CUDA si usas GPU)
- ESM de Evolutionary Scale (https://github.com/EvolutionaryScale/esm)
- pandas, tqdm

### 🧪 Instalación del entorno

```bash
conda create -n esm3 python=3.11
conda activate esm3

pip install torch pandas tqdm
pip install 'esm @ git+https://github.com/EvolutionaryScale/esm.git'
```

## 📥 Datos de entrada

El archivo CSV debe contener al menos una columna llamada `sequence`, con las secuencias de aminoácidos a embeber. En nuestro caso, también tiene un `DMS_ID` y un valor `DMS` (que en realidad representa el Kd) para su uso en los benchmarks trabajados por ProteinGyM.

Ejemplo:

```csv
sequence
ARNDCEQGHILKMFPSTWYV
AGPLMDKR...
...
```

## ▶️ Ejecución de scripts

### Paso 1: Extraer embeddings

El siguiente script realiza:

- Calcula el promedio de los embeddings de todos los aminoácidos de una secuencia.
- El resultado es un solo vector de tamaño [1536] por secuencia.
- Es una representación global de la proteína entera, capturando información de estructura, función y evolución.

```bash
python scripts/01_preprocces_extract_embeddings_avg.py
```
El siguiente script realiza lo siguiente:

* Extrae los **embeddings individuales para cada aminoácido** de la secuencia.
* No se promedian: se conserva la matriz [L, 1536] (L = longitud de la secuencia).
* Incluye señales combinadas de estructura, función, ubicación relativa, etc.
* Las dimensiones generadas serán: `token_map[seq] = emb  # shape: [seq_len, 1536]`

Esto generará archivos .pt en la carpeta embeddings_avg/, cada uno conteniendo un diccionario `{secuencia: vector_emb}`.

Si deseas guardar los embeddings por tokens individuales en lugar del promedio, ejecuta:

```bash
python scripts/01_preprocces_extract_embeddings_token.py
```

### Paso 2: Evaluar con zero-shot scoring

Este paso procesa las secuencias usando el modelo esm3-open y evalúa diferencias entre el aminoácido wild-type y el mutante, con base en el archivo de referencia `./reference_files/DMS_substitutions.csv`. Luego se comparan esos resultados con los scores experimentales (`DMS_score`, equivalente a Kd) mediante correlación de Spearman.

Para ejecutar:

```bash
python ./scripts/02_compute_zero_shot.py
```
Nota: El proceso se implementa en `./scripts/zero_shot/compute_fitness.py`, y `02_compute_zero_shot.py` se encarga de establecer los parámetros modificables según la base de datos o caso evaluado. Se puede modificar según el caso de análisis

## 💡 Notas útiles

- El modelo `esm3-open` se descargará automáticamente desde Hugging Face la primera vez que se use.
- Puedes ajustar el `batch_size` en los scripts del Paso 1 según tu GPU/CPU.
- Para cargar embeddings .pt:

```python
import torch
data = torch.load("path/to/{nombre_archivo}.pt")
```

## 📌 Créditos

- Modelo: `esm3-open` de Evolutionary Scale
- Scripts inspirados en benchmarks de: `ProteinGym`
- Proyecto personalizado por el equipo de investigación
