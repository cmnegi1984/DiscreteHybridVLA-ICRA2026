import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Norm stats from baseline norm_stats.json
STATE_MEAN = np.array([0.10899, -0.53329, -0.53725, -1.28309, 0.36701, 0.0,
                        0.47016, 0.0, 0.0, 0.0, 0.0, 0.00164,
                        -0.45295, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
STATE_STD  = np.array([0.15043, 0.49125, 0.64577, 0.50055, 0.60636, 1.0,
                        0.61056, 1.0, 1.0, 1.0, 1.0, 0.07585,
                        0.28268, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
STATE_STD  = np.where(STATE_STD == 0, 1.0, STATE_STD)

ACTION_MEAN = np.array([-0.000073, 0.000138, -0.000721, -0.000618, -0.000342, 0.0,
                          0.42135, 0.0, 0.0, 0.0, 0.0, 0.0000599,
                          -0.000404, 0.01479, 0.00105, -0.000848, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_STD  = np.array([0.02579, 0.04327, 0.03220, 0.02874, 0.05483, 1.0,
                          0.66724, 1.0, 1.0, 1.0, 1.0, 0.00535,
                          0.01296, 0.05166, 0.03492, 0.14005, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ACTION_STD  = np.where(ACTION_STD == 0, 1.0, ACTION_STD)


class VQCodebook(nn.Module):
    def __init__(self, K=512, d=8):
        super().__init__()
        self.codebook = nn.Embedding(K, d)
    def lookup(self, idx):
        return self.codebook(idx)


class DiscreteHybridVLA(nn.Module):
    def __init__(self, K=512, d=256, state_dim=8, action_dim=8):
        super().__init__()
        self.K = K
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.task_emb  = nn.Embedding(100, d)
        self.cls       = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 8, d*4, dropout=0.0, batch_first=True)
        self.fusion    = nn.TransformerEncoder(enc, 4)
        self.head      = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, K))
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, action_dim))
        self.codebook  = VQCodebook(K, action_dim)

    def predict(self, state_8dim, task_index=0):
        """state_8dim: numpy array shape (8,) — raw joint angles"""
        device = next(self.parameters()).device
        s = torch.tensor(state_8dim[:8], dtype=torch.float32).unsqueeze(0).to(device)
        t = torch.tensor([task_index], dtype=torch.long).to(device)
        x = torch.cat([self.cls.expand(1, -1, -1),
                        self.state_enc(s).unsqueeze(1),
                        self.task_emb(t).unsqueeze(1)], 1)
        z = self.head(self.fusion(x)[:, 0, :])
        idx    = z.argmax(-1)
        action = self.codebook.lookup(idx).squeeze(0).detach().cpu().numpy()
        return action  # shape (8,)

