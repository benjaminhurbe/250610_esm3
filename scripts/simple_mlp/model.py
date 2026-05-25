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

class Kdnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.Layer1 = nn.Linear(121*20, 300) 
    self.Layer2 = nn.Linear(300, 450)
    self.Layer3 = nn.Linear(450, 150)
    self.Layer4 = nn.Linear(150, 10)
    self.Layer5 = nn.Linear(10, 1)

  def forward(self, x):
    x =  F.relu(self.Layer1(x))
    x =  F.relu(self.Layer2(x))
    x =  F.relu(self.Layer3(x))
    z =  F.relu(self.Layer4(x))
    x =  self.Layer5(x)
    return x
