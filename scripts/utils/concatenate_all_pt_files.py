#!/usr/bin/env python3
import torch
import os
import argparse
from tqdm import tqdm 

def merge_pt_files(input_dir: str, output_path: str, overwrite: bool = False):
    """
    Recorre input_dir buscando todos los .pt, carga cada uno como dict y concatena
    sus elementos en un único diccionario que luego guarda en output_path.

    :param input_dir: Carpeta donde están los .pt
    :param output_path: Ruta (incluido nombre .pt) donde se guardará el merge
    :param overwrite: Si True, sobrescribe output_path si existe
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Ya existe {output_path}. Usa --overwrite para forzar.")

    merged = {}
    pt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pt")])

    for fname in tqdm(pt_files, desc="Uniendo .pt"):
        full = os.path.join(input_dir, fname)
        data = torch.load(full)
        # data debe ser un dict {seq: tensor}
        for seq, emb in data.items():
            if seq in merged:
                # Aviso en caso de duplicado, puedes cambiar a skip o reemplazar
                print(f"⚠️  Secuencia duplicada encontrada: {seq[:30]}...  Se está ignorando la segunda aparición.")
                continue
            merged[seq] = emb

    # Guarda el diccionario completo
    torch.save(merged, output_path)
    print(f"\n✅ Guardado {len(merged)} embeddings en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mergea múltiples archivos .pt de embeddings token-level en uno solo"
    )
    parser.add_argument("input_dir", help="Directorio con archivos .pt")
    #en este caso es   /media/nova/datos/diego/test/250610_esm3/results/embeddings_token 
    parser.add_argument("output_pt", help="Ruta de salida para el .pt combinado")
    # en este caso es   /media/nova/datos/diego/test/250610_esm3/results/embeddings_token/concatenated_embeddings_final.pt
    parser.add_argument(
        "--overwrite", "-f", action="store_true",
        help="Sobrescribir output_pt si ya existe"
    )
    args = parser.parse_args()

    merge_pt_files(args.input_dir, args.output_pt, args.overwrite)
