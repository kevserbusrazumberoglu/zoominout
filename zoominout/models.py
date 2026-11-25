# Residual adapter architectures.
# Extracted from the cell HÜCRE 10 of notebooks/ZoomInOut_patch_level.ipynb.
# An earlier version relearned a projection of the raw embedding from scratch
# and reduced 2048 to 512 dimensions, which risked damaging an already strong
# frozen representation with a small and noisy training signal; several
# variants indeed failed to beat the frozen model. The adapter instead learns a
# small correction to the raw embedding and its last layer is zero-initialised,
# so at the start of training the correction is exactly zero and the output
# equals the normalised frozen embedding. The untrained adapter is therefore
# identical to the frozen backbone, and training can only move away from that
# starting point.

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 2048
HIDDEN    = 1024
TEMP      = 0.07


class ResidualHead(nn.Module):
    # Raw embedding plus a small learnable correction. The last layer is
    # zero-initialised, so initially the correction is zero and the output is
    # normalize(raw_embedding).
    def __init__(self, embed_dim=EMBED_DIM, context_dim=0, hidden=HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embed_dim + context_dim, hidden), nn.GELU(),
                                  nn.LayerNorm(hidden), nn.Linear(hidden, embed_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, context=None):
        inp = x if context is None else torch.cat([x, context], dim=-1)
        return F.normalize(x + self.mlp(inp), dim=-1)


class ProjectionHead(nn.Module):
    # Unconditioned adapter, shared by the vanilla and the hard-negative arms.
    def __init__(self, input_dim=EMBED_DIM, hidden=HIDDEN):
        super().__init__()
        self.img_proj = ResidualHead(input_dim, context_dim=0, hidden=hidden)
        self.txt_proj = ResidualHead(input_dim, context_dim=0, hidden=hidden)

    def forward(self, img, txt):
        return self.img_proj(img), self.txt_proj(txt)

    def project_image(self, img):
        return self.img_proj(img)


class InstructionAwareHead(nn.Module):
    # Instruction-aware variant. It requires the oracle mode label and is used
    # only as an upper reference, not as the proposed model.
    def __init__(self, input_dim=EMBED_DIM, hidden=HIDDEN, num_modes=4):
        super().__init__()
        self.mode_emb = nn.Embedding(num_modes, input_dim)
        self.img_proj = ResidualHead(input_dim, context_dim=input_dim, hidden=hidden)
        self.txt_proj = ResidualHead(input_dim, context_dim=input_dim, hidden=hidden)

    def forward(self, img, txt, mode_idx):
        m = self.mode_emb(mode_idx)
        return self.img_proj(img, m), self.txt_proj(txt, m)

    def project_image(self, img, mode_idx):
        m = self.mode_emb(mode_idx)
        return self.img_proj(img, m)


class SelfInferringHead(nn.Module):
    # Self-inferring variant, which predicts the surface mode from the pair
    # itself and needs no oracle label. This is the proposed model.
    def __init__(self, input_dim=EMBED_DIM, hidden=HIDDEN, num_modes=4):
        super().__init__()
        self.mode_table = nn.Embedding(num_modes, input_dim)
        self.mode_predictor = nn.Sequential(nn.Linear(input_dim * 2, 256), nn.GELU(), nn.Linear(256, num_modes))
        self.img_proj = ResidualHead(input_dim, context_dim=input_dim, hidden=hidden)
        self.txt_proj = ResidualHead(input_dim, context_dim=input_dim, hidden=hidden)

    def infer_mode(self, img, txt):
        logits = self.mode_predictor(torch.cat([img, txt], dim=-1))
        probs = F.softmax(logits, dim=-1)
        mode_vec = probs @ self.mode_table.weight
        return mode_vec, logits

    def forward(self, img, txt):
        mode_vec, mode_logits = self.infer_mode(img, txt)
        z_img = self.img_proj(img, mode_vec)
        z_txt = self.txt_proj(txt, mode_vec)
        return z_img, z_txt, mode_logits

    def project_image_with_vec(self, img, mode_vec):
        return self.img_proj(img, mode_vec)
