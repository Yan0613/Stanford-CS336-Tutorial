import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, std=std, 
            a=-3.0 * std, b=3.0 * std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (..., in_features)
        # 返回: (..., out_features)
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...)        整数,每个元素在 [0, num_embeddings)
        # 返回:      (..., embedding_dim)   float
        return self.weight[token_ids]