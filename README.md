# 🧬 ESM3: Protein Embedding Processing Pipeline

This project enables the extraction of embeddings and the application of zero-shot analysis on a dataset of **65,000 mutant sequences** of the **CR9114 antigen**, together with an experimental **Kd binding value** to **Hemagglutinin subtype H1**.  
Embeddings are generated from amino acid sequences using the [`esm3-open`](https://huggingface.co/EvolutionaryScale/esm3-open) model by **Evolutionary Scale**, executed locally with GPU support if available.

---

## 📁 Project Structure


250610_esm3/

├── data/

│   └── A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv  # File containing amino acid sequences

├── DMS_ids/

│   └── múltiples bases de datos de proteingym (*.csv)      # DMS benchmark databases

├── reference_files/

│   └── clinical_substitutions.csv                          # ProteinGym reference CSV

│   └── clinical_indels.csv                                 # Contain the target protein

│   └── DMS_indels.csv                                      # DMS datasets of indels and substitutions

│   └── DMS_substitutions.csv                               # Integrated CR9114 dataset

├── scripts/

│   └── 01_preprocces_embeddings.py                         # Embedding extraction script


## ⚙️ Requirements

- Python 3.11 (recommended with Conda)  
- PyTorch (with CUDA support for GPU)  
- ESM by Evolutionary Scale ([GitHub link](https://github.com/EvolutionaryScale/esm))  
- pandas, tqdm  

---

### 🧪 Environment Setup

```bash
conda create -n esm3 python=3.11
conda activate esm3

pip install torch pandas tqdm
pip install 'esm @ git+https://github.com/EvolutionaryScale/esm.git'
```

## 📥 Input Data

The CSV file must contain at least one column named `sequence`, with the amino acid sequences to be embedded.  
In this project, it also includes a `DMS_ID` and a `DMS` value (which represents the Kd) for use in the benchmarks developed by ProteinGym.

Example:
```csv
sequence
ARNDCEQGHILKMFPSTWYV
AGPLMDKR...
...
```

## ▶️ Script Execution

### Step 1: Extract Embeddings

This script performs the following:

- Computes the **average of the embeddings** across all amino acids in a sequence.  
- The result is a single vector of size [1536] per sequence.  
- It provides a **global representation** of the entire protein, capturing information about structure, function, and evolution.


```bash
python scripts/01_preprocces_extract_embeddings_avg.py
```

The next script performs the following:

- Extracts the **individual embeddings for each amino acid** in the sequence.  
- They are **not averaged** — the resulting matrix is [L, 1536], where *L* is the sequence length.  
- These embeddings combine structural, functional, and positional signals.  
- The generated dimensions are: `token_map[seq] = emb  # shape: [seq_len, 1536]`.

This process generates `.pt` files inside the `embeddings_avg/` folder, each containing a dictionary of the form `{sequence: vector_emb}`.

If you prefer to save embeddings per token rather than averaged, run the corresponding script.

```bash
python scripts/01_preprocces_extract_embeddings_token.py
```
---

### Step 2: Evaluate with Zero-Shot Scoring

This step processes the sequences using the **esm3-open** model and evaluates differences between wild-type and mutant amino acids, based on the reference file `./reference_files/DMS_substitutions.csv`.  
Then, the results are compared with the experimental scores (`DMS_score`, equivalent to Kd) using **Spearman correlation**.

```bash
python ./scripts/02_compute_zero_shot.py
```
Nota: The implementation is contained in `./scripts/zero_shot/compute_fitness.py`, and the wrapper script `02_compute_zero_shot.py` manages customizable parameters depending on the dataset or experiment being evaluated.  
You can modify it according to the specific analysis case.


## 💡 Useful Notes

- The `esm3-open` model will be automatically downloaded from **Hugging Face** the first time it is used.  
- You can adjust the `batch_size` parameter in Step 1 scripts according to your GPU or CPU capacity.  
- To load `.pt` embeddings, simply use the corresponding PyTorch function.

```python
import torch
data = torch.load("path/to/{nombre_archivo}.pt")
```

## 📌 Credits

- Model: `esm3-open` by **Evolutionary Scale**  
- Scripts inspired by benchmarks from **ProteinGym**  
- Custom project developed by the **research team**
