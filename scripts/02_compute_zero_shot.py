#!/home/nova/anaconda3/envs/esm3/bin/python
"""
Wrapper para lanzar compute_fitness.py de proteingym sobre mi conjunto de DMS (Phillips) usando esm3-open (secuencia-only).
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Rutas
    project_root = Path(__file__).parent.parent
    script_cf   = project_root / "scripts" / "zero_shot" / "compute_fitness.py"
    reference   = "/media/nova/datos/diego/test/test_ad/250610_esm3/reference_files/DMS_substitutions.csv"
    dms_dir     = "/media/nova/datos/diego/test/test_ad/250610_esm3/DMS_ids"
    output_dir  = project_root / "results" / "compute_zero_shot_results"
    
    # Asegúrate de que exista la carpeta de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comando
    cmd = [
        sys.executable, str(script_cf),
        "--model_type",   "esm3_open",
        "--reference_csv", str(reference),
        "--dms_dir",       str(dms_dir),
        "--output_dir",    str(output_dir),
        "--DMS_index",     "218",
        # no usamos --use_structure para secuencia-only
    ]
    
    print("Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✓ Resultados en {output_dir}")

if __name__ == "__main__":
    main()
