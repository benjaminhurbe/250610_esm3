# ESM3 Protein Analysis Pipeline - Complete Process Overview

## Project Summary

This project implements a comprehensive pipeline for protein binding affinity (Kd) prediction using the ESM3 protein language model. The pipeline includes data preprocessing, embedding extraction, multiple training approaches (zero-shot and transfer learning), inference capabilities, and interactive web applications.

## 📁 Project Structure

```
250610_esm3/
├── data/                                    # Input datasets
├── model/                                   # Trained models and scalers
├── results/                                 # Output results and visualizations
├── scripts/                                 # Analysis and training scripts
│   ├── utils/                              # Utility functions
│   ├── graphics/                           # Visualization notebooks
│   └── zero_shot/                          # Zero-shot analysis
├── app.py                                   # Streamlit app (local ESM3)
└── app_pe.py                               # Streamlit app (pre-computed embeddings)
```

## 🔄 Complete Workflow

### Phase 1: Data Preprocessing and Embedding Extraction

#### 1.1 Average Embeddings Extraction
**Script:** `01_preprocces_extract_embeddings_avg.py`
- **Purpose:** Extract sequence-level average embeddings from ESM3-open model
- **Input:** CSV file with protein sequences and DMS scores
- **Process:**
  - Loads ESM3-open (1.4B parameters) model
  - Processes sequences in batches of 1000
  - Extracts token-level embeddings and computes mean across sequence length
  - Saves embeddings as PyTorch tensors (.pt files)
- **Output:** Batch files containing average embeddings per sequence

#### 1.2 Token-Level Embeddings Extraction
**Script:** `01_preprocces_extract_embeddings_token.py`
- **Purpose:** Extract position-specific token embeddings (no averaging)
- **Process:**
  - Uses identical ESM3-open configuration as average extraction
  - Preserves full sequence length dimension [seq_len, hidden_size]
  - Batch processing with proper sequence length handling
- **Output:** Token-level embeddings preserving positional information

#### 1.3 Embedding Consolidation
**Script:** `utils/concatenate_all_pt_files.py`
- **Purpose:** Merge multiple embedding batch files into single consolidated file
- **Process:**
  - Traverses embedding directories
  - Loads and merges all .pt files into single dictionary
  - Handles duplicate sequences with warnings
- **Output:** `concatenated_embeddings_final.pt` - single file with all embeddings

### Phase 2: Zero-Shot Analysis

#### 2.1 Zero-Shot Fitness Computation
**Script:** `02_compute_zero_shot.py`
- **Purpose:** Wrapper for ProteinGym zero-shot analysis
- **Process:**
  - Calls `zero_shot/compute_fitness.py` with ESM3-open model
  - Analyzes mutation effects using masked marginal probabilities
  - Computes correlation with experimental DMS scores
- **Output:** Zero-shot fitness predictions and correlations

#### 2.2 Core Zero-Shot Implementation
**Script:** `zero_shot/compute_fitness.py`
- **Purpose:** Core implementation of masked marginal approach
- **Features:**
  - Support for multiple ESM models (ESM-C, ESM3)
  - Windowing for long sequences
  - Structure-aware scoring (when PDB available)
  - Batch processing of mutations
  - Spearman correlation analysis

### Phase 3: Transfer Learning Approaches

#### 3.1 Initial Transfer Learning Implementation
**Script:** `03_transfer_learning_regression_head.py`
- **Purpose:** Production-ready transfer learning with regression head
- **Architecture:**
  - ESM3Regressor: Attention + MLP network
  - Light attention mechanism for sequence aggregation
  - 1536-dim input → 256-dim hidden → 1-dim output
- **Key Features:**
  - Comprehensive logging system
  - Memory management and optimization
  - Real-time performance monitoring
  - Automatic model and metrics saving
  - 95% speedup using pre-computed embeddings

#### 3.2 Transfer Learning Without Normalization
**Script:** `04_transfer_regress_without_norm.py`
- **Purpose:** Fine-tuning without target normalization
- **Process:**
  - Direct prediction of –log10(Kd) values
  - Batch size: 32, Epochs: 300
  - MSE loss function
  - Comprehensive metrics tracking (MSE, R², Spearman)
- **Output:** Model, predictions CSV, learning curves, scatter plots

#### 3.3 Transfer Learning With Normalization
**Script:** `05_transfer_regress_normalized.py`
- **Purpose:** Fine-tuning with StandardScaler normalization
- **Process:**
  - Normalizes target values to mean=0, std=1
  - Uses same architecture as non-normalized version
  - Saves scaler for inference denormalization
  - Comprehensive evaluation with original scale metrics
- **Output:** Normalized model, scaler, denormalized predictions and plots

#### 3.4 Model Architecture Definition
**Script:** `modelo_esm3_regresion.py`
- **Components:**
  - `LightAttention`: Attention mechanism for sequence aggregation
  - `ESM3Regressor`: Complete regression model
- **Architecture Details:**
  - Input: [B, L, D] embeddings
  - Attention: Learnable position weights with softmax
  - MLP: 1536 → 256 → 1 with ReLU, Dropout(0.2)
  - Output: Single Kd prediction per sequence

### Phase 4: Inference and Evaluation

#### 4.1 Model Inference
**Script:** `06_infer_esm3_regressor.py`
- **Purpose:** Inference with trained normalized model
- **Process:**
  - Loads pre-trained regressor and scaler
  - Processes example sequences from dataset
  - Generates predictions with confidence metrics
  - Creates scatter plots with identity lines
- **Output:** Inference results CSV and visualization plots

### Phase 5: Interactive Applications

#### 5.1 Local ESM3 Application
**Script:** `app.py`
- **Purpose:** Streamlit web interface using local ESM3 model
- **Features:**
  - Real-time embedding extraction using ESM3-open
  - Multiple prediction formats (training vs inference)
  - Interactive sequence input with examples
  - Kd interpretation (ultra-high to low affinity)
  - Technical debugging information
- **Use Case:** Fresh embedding extraction for new sequences

#### 5.2 Pre-computed Embeddings Application  
**Script:** `app_pe.py`
- **Purpose:** Streamlit interface using cached embeddings
- **Features:**
  - Utilizes pre-computed embedding dictionary
  - Identical results to inference script
  - Dataset sequence selection
  - Comparison with experimental values
  - Error analysis and interpretation
- **Use Case:** Fast inference on dataset sequences

### Phase 6: Visualization and Analysis

#### 6.1 Correlation Analysis
**Script:** `zero_shot/corr_plot.py`
- **Purpose:** Generate correlation plots for zero-shot analysis
- **Visualizations:**
  - Spearman correlation scatter plots
  - Performance comparison plots
  - Statistical significance analysis

#### 6.2 Embedding Visualization
**Notebooks:** `graphics/` directory
- **Content:**
  - t-SNE visualization of embeddings
  - Spearman correlation analysis
  - Performance comparison plots
  - Statistical analysis of predictions

## 🔧 Technical Configuration

### Model Specifications
- **ESM3-open:** 1.4B parameters
- **Embedding dimension:** 1536
- **Batch processing:** 32 sequences (training), 1000 sequences (embedding extraction)
- **Learning rate:** 1e-4
- **Loss function:** MSE

### Hardware Requirements
- **GPU:** CUDA-compatible for model inference
- **Memory:** Sufficient for 1.4B parameter model
- **Storage:** Several GB for embeddings and model checkpoints

## 📊 Performance Metrics

### Model Evaluation Metrics
- **MSE:** Mean Squared Error on original scale
- **R²:** Coefficient of determination
- **Spearman ρ:** Rank correlation coefficient
- **Pearson r:** Linear correlation coefficient

### Typical Results
- **Zero-shot:** Correlation varies by dataset (baseline)
- **Transfer learning:** Improved MSE and R² scores
- **Normalized models:** Better convergence and stability

## 🚀 Usage Instructions

### 1. Environment Setup
```bash
# Install required packages
pip install torch esm-models streamlit pandas scikit-learn matplotlib seaborn
```

### 2. Data Preparation
```bash
# Place your CSV file with 'sequence' and 'DMS_score' columns in data/
# Example: A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv
```

### 3. Run Complete Pipeline
```bash
# Step 1: Extract embeddings
python scripts/01_preprocces_extract_embeddings_token.py data/your_dataset.csv

# Step 2: Consolidate embeddings
python scripts/utils/concatenate_all_pt_files.py results/embeddings_token results/embeddings_token_final/concatenated_embeddings_final.pt

# Step 3: Zero-shot analysis
python scripts/02_compute_zero_shot.py

# Step 4: Transfer learning
python scripts/05_transfer_regress_normalized.py

# Step 5: Inference
python scripts/06_infer_esm3_regressor.py

# Step 6: Launch web app
streamlit run app_pe.py
```

## 📈 Key Innovations

### 1. **Performance Optimization**
- 95% speed improvement using pre-computed embeddings
- Efficient batch processing and memory management
- Windowing approach for long sequences

### 2. **Robust Architecture**
- Light attention mechanism for sequence aggregation
- Dropout regularization to prevent overfitting
- Flexible model architecture supporting various input formats

### 3. **Comprehensive Evaluation**
- Multiple correlation metrics (Spearman, Pearson, R²)
- Cross-validation and statistical significance testing
- Detailed visualization and error analysis

### 4. **Production-Ready Features**
- Extensive logging and monitoring
- Error handling and graceful degradation
- Model checkpointing and reproducibility
- Interactive web interfaces for accessibility

## 🔬 Scientific Context

This pipeline addresses the critical need for accurate protein binding affinity prediction in drug discovery and protein engineering. By combining:

1. **State-of-the-art protein language models** (ESM3) for representation learning
2. **Transfer learning approaches** for task-specific optimization  
3. **Comprehensive evaluation methodologies** for robust assessment
4. **Interactive tools** for practical deployment

The system provides a complete workflow from raw protein sequences to actionable binding affinity predictions, supporting both research and practical applications in computational biology.

## 📝 File Dependencies

### Core Dependencies
- `modelo_esm3_regresion.py` → Used by all training scripts
- `concatenated_embeddings_final.pt` → Required by transfer learning scripts
- `esm3_regressor_kd_normalized.pt` → Required by inference and apps
- `kd_scaler_normalized.joblib` → Required for normalized model inference

### Output Dependencies
- Training scripts → Generate models for inference
- Inference scripts → Generate predictions for evaluation
- Embedding extraction → Required for all downstream analysis

This comprehensive pipeline provides a complete solution for ESM3-based protein binding affinity prediction, from data preprocessing through interactive deployment.