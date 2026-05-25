# MAPA DE AMINOÁCIDOS - ÍNDICES

import torch
import torch.nn.functional as F

# Definir los 20 aminoácidos estándar
aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(aa_vocab)}

#TRANSFORMAR LA SECUENCIA A INDICES

seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQLR"  # aquí va tu secuencia de 120 aa

idxs = torch.tensor([aa_to_idx[aa] for aa in seq], dtype=torch.long)
print(idxs.shape)   # torch.Size([120])

#APLICAR ONE_HOT
one_hot = F.one_hot(idxs, num_classes=len(aa_vocab))
print(one_hot.shape)  # torch.Size([120, 20])
print(one_hot)