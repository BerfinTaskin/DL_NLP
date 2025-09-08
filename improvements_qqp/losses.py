import torch
import torch.nn.functional as F

def info_nce_logits(z_anchor, z_cands, temperature: float):
    """
    z_anchor: [B, d], z_cands: [C, d]  (C == B for SimCSE)
    returns logits [B, C] (cosine similarity / tau)
    """
    z_anchor = F.normalize(z_anchor, dim=1)
    z_cands  = F.normalize(z_cands,  dim=1)
    logits = torch.matmul(z_anchor, z_cands.T)  # cosine similarities
    return logits / temperature

def simcse_loss(z1, z2, temperature=0.05):
    """
    Supervised SimCSE: positives are diagonal pairs (i==i), negatives are other batch items.
    """
    logits = info_nce_logits(z1, z2, temperature)   # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)