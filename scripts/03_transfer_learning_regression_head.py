#!/usr/bin/env python
# coding: utf-8

# # Avance 4: Fine-tuning del modelo ESM3. 
# Explicación paso a paso
# 
# ## Generalidades: 
# En este cuaderno veremos como añadirle una capa extra al modelo ESM3, que nos permita afinarlo para que prediga directamente la Kd. En nuestra entrega previa, del desempeño del modelo zero-shot, el modelo botaba como output una probabilidad de ocurrencia de una mutación en su posición. A partir de ello, calculabamos una correlación con la Kd establecida experimentalmente para esa misma mutación. En este caso, ahora sí, buscamos, añadiendo una red neuronal por sobre el modelo ESM, que se prediga directamente la Kd, y no solamente una probabilidad.

# Primero, importamos librerías necesarias. 
# 
# Nótese que en la línea `from modelo_esm3_regresion import ESM3Regressor` estamos cargando nuestra propia red MLP+atención. Esta clase contiene la red MLP que entrenaremos por encima del modelo ESM3.

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from modelo_esm3_regresion import ESM3Regressor
import os
import time
import glob
#Para el logging
import logging
from datetime import datetime

# Setup simple para el logging
log_filename = f"esm3_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def log_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    logging.info(message)
    print(*args, **kwargs)

#Inicio verdadero del notebook
log_print("=== Iniciando entrenamiento de ESM3 con regresión ===")
# A continuación definimos ubiaciones e hiperparámetros, como el tamaño de lote, las épocas, learning rate y dimensión de embedding. Recordemos que para el modelo ESM3 que estamos usando, de 1.4 B de parámetros, el tamaño del embedding es de 1536

# In[2]:


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_print(f"Using device: {DEVICE}")
CSV_PATH = "../data/A0A1K4LHP2_CR9114_Phillips_2021_updated_target.csv"
MODEL_SAVE_PATH = "../model/esm3_regressor_kd.pt"
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
USE_ATTENTION = True
EMBED_DIM = 1536


# Enseguida, definimos nuestra clase ProteinDataset, que aceptará un dataframe, para importar a partir de un índice nuestra secuencia y su valor objetivo. Esto es costumbre para cada dataset particular sobre el cual se usará Pytorch

# In[3]:


# === Dataset personalizado ===
class ProteinDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["sequence"], torch.tensor(row["kd"], dtype=torch.float32)


# Leemos nuestra base de datos, obtenemos las dos columnas de interés: secuencia y Kd. Tenemos 65 093 filas que servirán de entrenamiento. Asimismo, usaremos el 80% de las secuencias para entrenar, y el 20% como dataset de prueba

# In[4]:


# === Cargar datos ===
df = pd.read_csv(CSV_PATH)[["sequence", "DMS_score"]].dropna().rename(columns={"DMS_score": "kd"})
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
log_print(f"Dataframe: {df}")
log_print("------------------------------------")
log_print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


# Cargamos nuestros dataframes de entrenamiento y prueba en nuestra clase ProteinDataset, y vemos una muestra.
# 
# A continuación, cargamos en DatasetLoader para que se nos devuelva la información en BATCHES

# In[5]:


BATCH_SIZE = 1
train_dataset = ProteinDataset(train_df)
test_dataset = ProteinDataset(test_df)
log_print(train_dataset)
log_print(f"Train dataset size: {len(train_dataset)}")
log_print("Train dataset sample: Sequence -------------- Kd")
log_print(train_dataset[2])
log_print("------------------------------------")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
log_print(f"Train loader size: {len(train_loader)}")
log_print("Train loader sample: Sequence -------------- Kd")
for seq, kd in train_loader:
    log_print(seq, "--------------", kd)
    break


# Cargamos modelo ESM3 preentrenado en modo evaluación. 

# In[6]:


log_print(DEVICE)
esm3_model = ESM3.from_pretrained("esm3-open").to("cuda:0")
esm3_model.eval()


# Programamos la configuración default del modelo para extraer solo los embeddings de cada posición (token) de la secuencia que solicitemos

# In[7]:


logits_config = LogitsConfig(
    sequence=True,
    return_embeddings=True,
    return_hidden_states=False)


# Instanciamos nuestra clase ESM3Regressor, y lo replicamos en ambas GPUs disponibles. Definimos el optimizador Adam sobre todos los parámetros entrenables y establecemos nuestra función de pérdida como el MSE.

# In[8]:


regressor = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to("cuda:0")
#regressor = nn.DataParallel(regressor, device_ids=[0,1])
optimizer = torch.optim.Adam(regressor.parameters(), lr=LR)
loss_fn = nn.MSELoss()


# Creamos una función para codificar cada secuencia y obtener su embedding final completo

# In[9]:


def embed_sequence(sequence):
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        inputs = esm3_model.encode(protein).to(DEVICE)
        output = esm3_model.logits(inputs, logits_config)
        return output.embeddings.squeeze(0).to(DEVICE)  # [seq_len, embed_dim=1536]


# Nuetro entrenamiento consiste en un numero determinado de épocas, y cada época es una pasada completa por los datos de entrenamiento.
# 
# Pasos seguidos en cada época:
# 
#     1. Extraemos embeddings de cada secuencia con nuestra función embed_sequence. Este, como se verá más adelante, es un paso que nos toma demasiado tiempo.
# 
#     2. Predecimos la Kd y hallamos la pérdida en base a nuestra loss function que es MSE
# 
#     3. Reseteo de gradientes, propagación de pérdida por la red (backward()) llenando .grad en cada parámetro entrenable. Luego al aplicar step() el optimizador Adam actualiza los parámetros según los gradientes, y la tasa de aprendizaje
# 
#     4. Acumulo el valor del tensor de pérdida en total_loss
# 
#     5. Al terminar todos los batches de una época, imprimo la pérdida media
# 
#     6. Para cada muestra en test_loader recogemos su predicción y el valor real, para luego obtener el MSE y R2 de evaluación

# En esta sección de entrenamiento, primero aplicamos UNA sola época, y vemos qué parte del entrenamiento toma más tiempo. Para ello, aplicamos timers tanto en la parte de la obtención del embedding de la secuencia, como en la predicción de nuestro MLP regressor, y en la sección de actualización de los pesos del regresor.

# In[10]:


train_losses = []
test_losses  = []
epoch_times = []
embed_times = []
regressor_times = []
optimizer_times = []
EPOCHS=1 #Ver qué demora más en una época
for epoch in range(1, EPOCHS + 1):
    start_epoch = time.perf_counter()
    regressor.train()
    total_loss = 0
    count = 0

    for seq, kd in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        seq = seq[0]
        kd = kd.to(DEVICE)
        if count > 300:
            # Salimos del bucle de batches cuando alcancemos 500
            break

        try:
            # ======= Medición de tiempo para embed_sequence =======
            start_embed = time.perf_counter()
            emb = embed_sequence(seq).unsqueeze(0)  # [1, seq_len, embed_dim]
            end_embed = time.perf_counter()
            embed_time = end_embed - start_embed
            embed_times.append(embed_time)

            # ======= Medición de tiempo para regresor =======
            start_regressor = time.perf_counter()
            pred = regressor(emb)
            loss = loss_fn(pred, kd)
            end_regressor = time.perf_counter()
            regressor_time = end_regressor - start_regressor
            regressor_times.append(regressor_time)

            # ======= Medición de tiempo para optimización =======
            start_optim = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_optim = time.perf_counter()
            optim_time = end_optim - start_optim
            optimizer_times.append(optim_time)

            total_loss += loss.item()
            count += 1

            # ======= Reporte cada 50 batches =======
            if count % 75 == 0:
                avg_embed = sum(embed_times[-50:]) / 50
                avg_regressor = sum(regressor_times[-50:]) / 50
                avg_optim = sum(optimizer_times[-50:]) / 50
                total_batch = avg_embed + avg_regressor + avg_optim

                log_print(f"\n⏱️ Batch {count} | Tiempos (últimos 50 batches):")
                log_print(f"  ├─ Embed: {avg_embed:.4f}s ({avg_embed/total_batch*100:.1f}%)")
                log_print(f"  ├─ Regressor: {avg_regressor:.4f}s ({avg_regressor/total_batch*100:.1f}%)")
                log_print(f"  └─ Optimización: {avg_optim:.4f}s ({avg_optim/total_batch*100:.1f}%)")
                log_print(f"  Total/batch: {total_batch:.4f}s | Velocidad: {50/total_batch:.2f} batch/s")

        except Exception as e:
            log_print(f"Error en entrenamiento: {e}")
            continue

    # ======= Final de época =======
    avg_train_loss = total_loss / count
    log_print(f"\n✓ Epoch {epoch} | Train Loss promedio: {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)

    # === Evaluación ===
    regressor.eval()
    preds, targets = [], []
    with torch.no_grad():
        count2=0
        for seq, kd in tqdm(test_loader, desc=f"Epoch {epoch} [Eval]"):
            seq = seq[0]  
            kd = kd.to(DEVICE)
            count2+=1
            if count2 == 200:
                # Salimos del bucle de batches cuando alcancemos 500
                log_print("Test evaluation limit reached (200 secuencias!!).")
                break
            try:
                emb = embed_sequence(seq).unsqueeze(0)
                #log_print(f"Embedding shape: {emb.shape}")
                pred = regressor(emb)
                preds.append(pred.item())
                targets.append(kd.item())
                #log_print(f"Pred shape: {pred.shape}, kd shape: {kd.shape}")
            except Exception as e:
                log_print(f"Error en evaluación: {e}")
                continue

    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    log_print(f"🎯 Epoch {epoch} | Test MSE: {mse:.4f} | R²: {r2:.4f}")
    test_losses.append(mse)

    end_epoch = time.perf_counter()
    duration = end_epoch - start_epoch
    epoch_times.append(duration)

    # ======= Reporte de tiempos de época =======
    avg_embed_epoch = sum(embed_times[-count:]) / count if count > 0 else 0
    avg_regressor_epoch = sum(regressor_times[-count:]) / count if count > 0 else 0
    avg_optim_epoch = sum(optimizer_times[-count:]) / count if count > 0 else 0

    log_print(f"\n📊 RESUMEN TIEMPOS Epoch {epoch}")
    log_print(f"  ├─ Embed: {avg_embed_epoch:.4f}s/batch ({len(embed_times)} batches)")
    log_print(f"  ├─ Regressor: {avg_regressor_epoch:.4f}s/batch")
    log_print(f"  ├─ Optimización: {avg_optim_epoch:.4f}s/batch")
    log_print(f"  └─ Total época: {duration:.1f}s ({count/duration:.2f} batch/s)")
    log_print(f"⏱️ Epoch {epoch} duración: {duration:.1f}s")


# Truncamos la salida, y vemos que, para cada secuencia, se toma el 90% del tiempo solamente en obtener su embedding. Pero nosotros ya tenemos calculado los embeddings de cada secuencia, gracias a nuestro trabajo de preprocesamiento. Así que, a continuación, simplemente importamos el archivo con los embeddings y los cargamos a un diccionario

# In[11]:


# === Cargar embeddings desde el archivo concatenado ===
EMBEDDINGS_DIR = "/media/nova/datos/proj_esm3_proof/250610_esm3/results/embeddings_token_final"
CONCATENATED_PATH = os.path.join(EMBEDDINGS_DIR, "concatenated_embeddings_final.pt")
EMBEDDINGS_DICT = {}

log_print("Cargando embeddings precalculados...")
start_load = time.time()

if os.path.exists(CONCATENATED_PATH):
    try:
        EMBEDDINGS_DICT = torch.load(CONCATENATED_PATH, map_location='cpu')
        log_print(f"✓ Embeddings cargados desde archivo concatenado: {len(EMBEDDINGS_DICT)} secuencias")

    except Exception as e:
        log_print(f"Error al cargar el archivo concatenado: {e}")
else:
    log_print(f"Error: Concatenated embeddings file not found at {CONCATENATED_PATH}")
log_print(f"Tiempo de carga: {time.time() - start_load:.2f} segundos")
log_print(f"Memoria usada: {len(EMBEDDINGS_DICT) * EMBED_DIM * 4 / 1e9:.2f} GB (aproximado)")


# In[12]:


#count = 0
#for seq_id, emb in EMBEDDINGS_DICT.items():
#    count += 1
#    EMBEDDINGS_DICT[seq_id] = emb.to("cuda:0")
#    log_print(count)
    #log_print(f"✓ Embeddings cargados y movidos a {DEVICE}")
#log_print(count)


# In[13]:


log_print(f"✓ Embeddings cargados: {len(EMBEDDINGS_DICT)} secuencias")
for idx, (key, value) in enumerate(EMBEDDINGS_DICT.items()):
    if idx < 5:  # Mostrar solo los primeros 5 embeddings
        log_print(f"{key}: {value.shape}")
    else:
        break
problem_sequence = 'QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGSTAYAQKFQGRVTITADKSTNTAYMELSSLRSEDTAVYYCARHGNYYYYYGMDVWGQGTTVTVSS'
log_print(problem_sequence in EMBEDDINGS_DICT)  # Debería ser True


# Nuevo entrenamiento, pero esta vez como usaremos los embeddings precalculados, aprovechamos y reestablecemos el tamaño del batch size a 32, porque esta vez se irá mucho más rápido.

# In[14]:


BATCH_SIZE=32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
log_print(f"Train loader size: {len(train_loader)}")
log_print("Train loader sample: Sequence -------------- Kd")
for seq, kd in train_loader:
    log_print(seq, "--------------", kd)
    break


# Instanciamos un nuevo Regressor

# In[15]:


regressor = ESM3Regressor(input_dim=EMBED_DIM, use_attention=USE_ATTENTION).to("cuda:0")
#regressor = nn.DataParallel(regressor, device_ids=[0,1])
optimizer = torch.optim.Adam(regressor.parameters(), lr=LR)
loss_fn = nn.MSELoss()


# Actualizamos nuestra función de recuperación de embeddings para usar directamente el diccionario con los embeddings precalculados

# In[16]:


def embed_batch_from_cache(seq_ids):
    """
    Process a batch of sequences. 
    """
    embeddings = [EMBEDDINGS_DICT[seq_id] for seq_id in seq_ids]
    return torch.stack(embeddings).to(DEVICE)  # Aseguramos que las dimensiones sean correctas


# Resumen de parametros entrenables de nuestro regresor

# In[17]:


for name, param in regressor.named_parameters():
    if param.requires_grad:
        log_print(f"{name:30} | {param.numel():>8} parámetros")


# Ahora sí, entrenamos nuevamente, y vemos que el porcentaje de tiempo usado en obtener los embeddings, y también su valor absoluto, se reduce dramáticamente, incrementando la rapidez del entrenamiento de nuestra capa de regresión

# In[18]:


train_losses = []
test_losses  = []
epoch_times = []
embed_times = []
r2_scores = []
regressor_times = []
optimizer_times = []
EPOCHS=200 #Ver qué demora más en una época
for epoch in range(1, EPOCHS + 1):
    start_epoch = time.perf_counter()
    regressor.train()
    total_loss = 0
    count = 0
    #batch_counter = 0  # Contador de batches

    for seq, kd in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        seq = list(seq)  # Convierte el batch de secuencias a lista
        kd = kd.to(DEVICE)
        #log_print(f"Batch size: {len(seq)}")
        #batch_counter += 1
        #if batch_counter > 10:
            # Salimos del bucle de batches cuando alcancemos 500
        #    break
        try:
            # ======= Medición de tiempo para embed_sequence =======
            start_embed = time.perf_counter()
            emb = embed_batch_from_cache(seq)  # [1, seq_len, embed_dim]
            #log_print(f"Embedding shape: {emb.shape}")
            #log_print(emb.shape)
            end_embed = time.perf_counter()
            embed_time = end_embed - start_embed
            embed_times.append(embed_time)

            # ======= Medición de tiempo para regresor =======
            start_regressor = time.perf_counter()
            pred = regressor(emb)  # [1, embed_dim]
            #log_print(f"Pred shape: {pred.shape}, kd shape: {kd.shape}")
            loss = loss_fn(pred, kd)
            #log_print(f"Loss: {loss.item()}")
            end_regressor = time.perf_counter()
            regressor_time = end_regressor - start_regressor
            regressor_times.append(regressor_time)
            #log_print(f"Regressor time: {regressor_time:.4f}s")

            # ======= Medición de tiempo para optimización =======
            start_optim = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_optim = time.perf_counter()
            optim_time = end_optim - start_optim
            optimizer_times.append(optim_time)

            total_loss += loss.item()
            count += 1

            # ======= Reporte a los 500 batches =======
            if count == 500:
                avg_embed = sum(embed_times[-50:]) / 50
                avg_regressor = sum(regressor_times[-50:]) / 50
                avg_optim = sum(optimizer_times[-50:]) / 50
                total_batch = avg_embed + avg_regressor + avg_optim

                log_print(f"\n⏱️ Batch {count} | Tiempos (últimos 50 batches):")
                log_print(f"  ├─ Embed: {avg_embed:.4f}s ({avg_embed/total_batch*100:.1f}%)")
                log_print(f"  ├─ Regressor: {avg_regressor:.4f}s ({avg_regressor/total_batch*100:.1f}%)")
                log_print(f"  └─ Optimización: {avg_optim:.4f}s ({avg_optim/total_batch*100:.1f}%)")
                log_print(f"  Total/batch: {total_batch:.4f}s | Velocidad: {50/total_batch:.2f} batch/s")

        except Exception as e:
            log_print(f"Error en entrenamiento: {e}")
            continue

    # ======= Final de época =======
    avg_train_loss = total_loss / count
    log_print(f"\n✓ Epoch {epoch} | Train Loss promedio: {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)

    # === Evaluación ===
    regressor.eval()
    preds, targets = [], []
    with torch.no_grad():
        #count3 = 0
        for seq, kd in tqdm(test_loader, desc=f"Epoch {epoch} [Eval]"):
            #count3 += 1
            #if count3 > 100:
                # Salimos del bucle de batches cuando alcancemos 500
            #    break
            seq = list(seq)            
            kd = kd.to(DEVICE)
            try:
                emb = embed_batch_from_cache(seq)
                #log_print(f"Embedding shape: {emb.shape}")
                pred = regressor(emb)  # [1, embed_dim]
                #log_print(f"Pred shape: {pred.shape}, kd shape: {kd.shape}")
                preds.extend(pred.cpu().tolist())
                targets.extend(kd.cpu().tolist())
                #log_print(f"Pred shape: {pred.shape}, kd shape: {kd.shape}")
            except Exception as e:
                log_print(f"Error en evaluación: {e}")
                continue

    mse = mean_squared_error(targets, preds)
    log_print(f"Preds: {len(preds)}, Targets: {len(targets)}")
    log_print(f"Preds: {preds[:5]}, Targets: {targets[:5]}")
    log_print(mse)
    r2 = r2_score(targets, preds)
    log_print(r2)
    log_print(f"🎯 Epoch {epoch} | Test MSE: {mse:.4f} | R²: {r2:.4f}")
    test_losses.append(mse)
    r2_scores.append(r2)  # <--- AÑADE ESTA LÍNEA

    end_epoch = time.perf_counter()
    duration = end_epoch - start_epoch
    epoch_times.append(duration)

    # ======= Reporte de tiempos de época =======
    avg_embed_epoch = sum(embed_times[-count:]) / count if count > 0 else 0
    avg_regressor_epoch = sum(regressor_times[-count:]) / count if count > 0 else 0
    avg_optim_epoch = sum(optimizer_times[-count:]) / count if count > 0 else 0
    if epoch <5:
        log_print(f"\n📊 RESUMEN TIEMPOS Epoch {epoch}")
        log_print(f"  ├─ Embed: {avg_embed_epoch:.4f}s/batch ({len(embed_times)} batches)")
        log_print(f"  ├─ Regressor: {avg_regressor_epoch:.4f}s/batch")
        log_print(f"  ├─ Optimización: {avg_optim_epoch:.4f}s/batch")
        log_print(f"  └─ Total época: {duration:.1f}s ({count/duration:.2f} batch/s)")
        log_print(f"⏱️ Epoch {epoch} duración: {duration:.1f}s")


# Graficamos la evolución de las pérdidas en entrenamiento y test

# In[ ]:


log_print(f"shape train losses: {len(train_losses)}")
log_print(f"shape train losses: {len(test_losses)}")
log_print(f"Total epochs: {EPOCHS}")


# In[ ]:


import matplotlib.pyplot as plt
plot_dir = "../results/transfer_learning_regression_head/"
epochs = list(range(1, EPOCHS+1))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses,  label='Test  MSE')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Curvas de aprendizaje')
plt.savefig(os.path.join(plot_dir, 'curvas_aprendizaje.png'), dpi=300, bbox_inches='tight')
#plt.show()


# In[ ]:


r2_scores
plt.plot(epochs, r2_scores, label='R² Score')


# In[ ]:

# === Encontrar el mejor modelo ===
best_epoch = r2_scores.index(max(r2_scores)) + 1  # +1 porque las épocas empiezan en 1
best_r2 = max(r2_scores)
best_mse = test_losses[best_epoch - 1]  # -1 porque las listas empiezan en 0

log_print(f"🌟 Mejor modelo: Época {best_epoch} con R² = {best_r2:.4f} y MSE = {best_mse:.4f}")

# === Guardar métricas y modelo ===
torch.save({
    'train_losses': train_losses,
    'test_losses': test_losses,
    'r2_scores': r2_scores,
    'epoch_times': epoch_times,
    'best_epoch': best_epoch,
    'best_r2': best_r2,
    'best_mse': best_mse,
    'model_state': regressor.state_dict(),
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LR,
        'embed_dim': EMBED_DIM,
        'use_attention': USE_ATTENTION
    }
}, "../model/training_metrics.pt")

# === Guardar solo el modelo ===
torch.save(regressor.state_dict(), MODEL_SAVE_PATH)

log_print(f"✅ Modelo final guardado en {MODEL_SAVE_PATH}")
log_print(f"✅ Métricas de entrenamiento guardadas en ../model/training_metrics.pt")
log_print(f"🌟 Mejor modelo: Época {best_epoch} con R² = {best_r2:.4f}")

# === Crear directorio si no existe ===
import os
os.makedirs(plot_dir, exist_ok=True)

# === Graficar resultados ===
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
epochs = list(range(1, EPOCHS+1))
plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(epochs, test_losses, 'r-', label='Test MSE', linewidth=2)
plt.plot(best_epoch, best_mse, 'ro', markersize=8, label=f'Mejor MSE: {best_mse:.4f}')
plt.title('Evolución de Pérdidas', fontsize=14)
plt.ylabel('Pérdida', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(epochs, r2_scores, 'g-', label='R² Score', linewidth=2)
plt.plot(best_epoch, best_r2, 'ro', markersize=8, label=f'Mejor R²: {best_r2:.4f}')
plt.title('Evolución del R²', fontsize=14)
plt.xlabel('Época', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()  # Cierra la figura para liberar memoria

log_print(f"📊 Gráfico guardado en {plot_dir}training_metrics.png")
