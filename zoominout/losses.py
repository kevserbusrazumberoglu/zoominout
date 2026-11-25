# Contrastive losses.
# Extracted from the cell HÜCRE 11 of notebooks/ZoomInOut_patch_level.ipynb.
# The hard negative is not a separate loss term: it is one extra column added
# to the text-to-image logits, masked out for the pairs that have none.

import torch
import torch.nn.functional as F

from .models import TEMP

def plain_info_nce(img, txt, **kwargs):
    batch_size = img.size(0)
    logits = img @ txt.T / TEMP
    labels = torch.arange(batch_size, device=img.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def hn_info_nce(img, txt, z_hn, has_hn, **kwargs):
    batch_size = img.size(0)
    labels = torch.arange(batch_size, device=img.device)
    logits_i2t = img @ txt.T / TEMP
    loss_i2t = F.cross_entropy(logits_i2t, labels)
    logits_t2i_base = txt @ img.T / TEMP
    # When a batch contains no hard negative at all, z_hn arrives as None. In
    # that case the extra column is skipped and the loss falls back to plain
    # InfoNCE.
    if z_hn is not None:
        sim_txt_hn = (txt * z_hn).sum(-1, keepdim=True) / TEMP
        sim_txt_hn = sim_txt_hn.masked_fill((has_hn.unsqueeze(1) == 0), float("-inf"))
        logits_t2i = torch.cat([logits_t2i_base, sim_txt_hn], dim=1)
    else:
        logits_t2i = logits_t2i_base
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    return (loss_i2t + loss_t2i) / 2


def hn_info_nce_realneg(img, txt, z_hn, has_hn, z_chn=None, has_chn=None, **kwargs):
    # Extension of hn_info_nce that adds, next to the surface hard negative
    # z_hn, a second column for a semantic hard negative obtained from the COCO
    # category annotations. With z_chn and has_chn omitted the behaviour is
    # identical to hn_info_nce.
    batch_size = img.size(0)
    labels = torch.arange(batch_size, device=img.device)
    logits_i2t = img @ txt.T / TEMP
    loss_i2t = F.cross_entropy(logits_i2t, labels)
    logits_t2i_base = txt @ img.T / TEMP
    extra_cols = [logits_t2i_base]
    if z_hn is not None:
        sim_txt_hn = (txt * z_hn).sum(-1, keepdim=True) / TEMP
        sim_txt_hn = sim_txt_hn.masked_fill((has_hn.unsqueeze(1) == 0), float("-inf"))
        extra_cols.append(sim_txt_hn)
    if z_chn is not None and has_chn is not None:
        sim_txt_chn = (txt * z_chn).sum(-1, keepdim=True) / TEMP
        sim_txt_chn = sim_txt_chn.masked_fill((has_chn.unsqueeze(1) == 0), float("-inf"))
        extra_cols.append(sim_txt_chn)
    logits_t2i = torch.cat(extra_cols, dim=1)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    return (loss_i2t + loss_t2i) / 2


MODE_MAP = {"semantic": 0, "color": 1, "shape": 2, "texture": 3}
# Selected by a sweep: 0.1 gives the best MRR (0.363), while values of 2.0 and
# above let the auxiliary mode term dominate and reduce R@1.
MODE_AUX_WEIGHT = 0.1
