# Chain-distance loss, used only for the ablation that tests the sequence
# structure as an explicit loss term. The proposed configuration does not
# include it.
# Extracted from the cell HÜCRE 14 of notebooks/ZoomInOut_patch_level.ipynb.

import torch
import torch.nn.functional as F

class ChainBatch:
    def __init__(self, emb, modality, mask, step_idx):
        self.emb = emb
        self.modality = modality
        self.mask = mask
        self.step_idx = step_idx

    def to(self, device):
        return ChainBatch(self.emb.to(device), self.modality.to(device),
                           self.mask.to(device), self.step_idx.to(device))


def project_chain_batch(model, batch: "ChainBatch"):
    B, L, D = batch.emb.shape
    flat = batch.emb.view(B * L, D)
    z_img = F.normalize(model.img_proj(flat), dim=-1)
    z_txt = F.normalize(model.txt_proj(flat), dim=-1)
    mod = batch.modality.view(B * L, 1).float()
    z = z_img * (1 - mod) + z_txt * mod
    return z.view(B, L, -1)


def sequence_aware_loss(z, batch: "ChainBatch", temp=0.07, pos_window=4,
                         lambda_multipos=0.5, lambda_rank=0.5, margin=0.1,
                         max_rank_samples_per_chain=6, same_modality_only=False):
    # With same_modality_only=False the multipos and rank terms ignore
    # modality, so an image node and a text node inside the same window both
    # count as positives. In the lambda sweep this variant did not beat the
    # baseline at any setting: the alignment term already teaches the
    # cross-modal correspondence, and a modality-blind term adds a conflicting
    # signal to it.
    # With same_modality_only=True both terms are computed only between nodes
    # of the same modality, image against image and text against text, so the
    # intra-modal structure is learned without competing with the cross-modal
    # objective.
    B, L, Dp = z.shape
    device = z.device

    flat_z, flat_chain, flat_step, flat_mod = [], [], [], []
    for b in range(B):
        valid = batch.mask[b]
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        flat_z.append(z[b, valid])
        flat_chain.append(torch.full((n_valid,), b, device=device, dtype=torch.long))
        flat_step.append(batch.step_idx[b, valid])
        flat_mod.append(batch.modality[b, valid])
    flat_z = torch.cat(flat_z, dim=0)
    flat_chain = torch.cat(flat_chain, dim=0)
    flat_step = torch.cat(flat_step, dim=0)
    flat_mod = torch.cat(flat_mod, dim=0)
    N = flat_z.size(0)

    sim = flat_z @ flat_z.T / temp
    same_chain = flat_chain.unsqueeze(0) == flat_chain.unsqueeze(1)
    same_modality = flat_mod.unsqueeze(0) == flat_mod.unsqueeze(1)
    step_dist = (flat_step.unsqueeze(0) - flat_step.unsqueeze(1)).abs()
    is_self = torch.eye(N, dtype=torch.bool, device=device)

    pos_mask = same_chain & (step_dist > 0) & (step_dist <= pos_window) & (~is_self)
    if same_modality_only:
        pos_mask = pos_mask & same_modality
    logits = sim.masked_fill(is_self, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = log_prob.masked_fill(is_self, 0.0)
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_counts
    has_pos = pos_mask.sum(dim=1) > 0
    l_multipos = per_anchor[has_pos].mean() if has_pos.any() else torch.zeros((), device=device)

    rank_terms = []
    for b in range(B):
        idx_b = (flat_chain == b).nonzero(as_tuple=True)[0]
        if idx_b.numel() < 3:
            continue
        n_sample = min(max_rank_samples_per_chain, idx_b.numel())
        perm = idx_b[torch.randperm(idx_b.numel(), device=device)[:n_sample]]
        for a in perm:
            others = idx_b[idx_b != a]
            if same_modality_only:
                others = others[flat_mod[others] == flat_mod[a]]
            if others.numel() < 2:
                continue
            d = (flat_step[others] - flat_step[a]).abs()
            order = torch.argsort(d)
            near, far = others[order[0]], others[order[-1]]
            if d[order[-1]] <= d[order[0]]:
                continue
            s_near = (flat_z[a] * flat_z[near]).sum() / temp
            s_far = (flat_z[a] * flat_z[far]).sum() / temp
            rank_terms.append(F.relu(torch.tensor(margin, device=device) - (s_near - s_far)))
    l_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=device)

    return lambda_multipos * l_multipos + lambda_rank * l_rank, {
        "l_multipos": l_multipos.item(), "l_rank": l_rank.item(),
        "n_pos_anchors": int(has_pos.sum().item()), "n_rank_triplets": len(rank_terms),
    }


def sequence_aware_loss_continuous(z, batch: "ChainBatch", temp=0.07, tau=3.0,
                                    lambda_multipos=0.5, lambda_rank=0.5, margin=0.1,
                                    max_rank_samples_per_chain=6, same_modality_only=True,
                                    max_weight_dist=None):
    # Continuous variant: instead of the hard pos_window threshold, every pair
    # within a chain contributes with an exponentially decaying weight,
    # weight(i, j) = exp(-step_dist(i, j) / tau). Near steps contribute much
    # and distant ones little, which replaces the binary positive decision with
    # a continuous distance signal. Only the multipos term changed; the rank
    # term is the same as in sequence_aware_loss.
    B, L, Dp = z.shape
    device = z.device

    flat_z, flat_chain, flat_step, flat_mod = [], [], [], []
    for b in range(B):
        valid = batch.mask[b]
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        flat_z.append(z[b, valid])
        flat_chain.append(torch.full((n_valid,), b, device=device, dtype=torch.long))
        flat_step.append(batch.step_idx[b, valid])
        flat_mod.append(batch.modality[b, valid])
    flat_z = torch.cat(flat_z, dim=0)
    flat_chain = torch.cat(flat_chain, dim=0)
    flat_step = torch.cat(flat_step, dim=0)
    flat_mod = torch.cat(flat_mod, dim=0)
    N = flat_z.size(0)

    sim = flat_z @ flat_z.T / temp
    same_chain = flat_chain.unsqueeze(0) == flat_chain.unsqueeze(1)
    same_modality = flat_mod.unsqueeze(0) == flat_mod.unsqueeze(1)
    step_dist = (flat_step.unsqueeze(0) - flat_step.unsqueeze(1)).abs().float()
    is_self = torch.eye(N, dtype=torch.bool, device=device)

    eligible = same_chain & (~is_self)
    if same_modality_only:
        eligible = eligible & same_modality
    weight = torch.exp(-step_dist / tau) * eligible.float()
    if max_weight_dist is not None:
        weight = weight * (step_dist <= max_weight_dist).float()

    logits = sim.masked_fill(is_self, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = log_prob.masked_fill(is_self, 0.0)

    weight_sum = weight.sum(dim=1).clamp(min=1e-8)
    per_anchor = -(log_prob * weight).sum(dim=1) / weight_sum
    has_pos = weight.sum(dim=1) > 1e-6
    l_multipos = per_anchor[has_pos].mean() if has_pos.any() else torch.zeros((), device=device)

    rank_terms = []
    for b in range(B):
        idx_b = (flat_chain == b).nonzero(as_tuple=True)[0]
        if idx_b.numel() < 3:
            continue
        n_sample = min(max_rank_samples_per_chain, idx_b.numel())
        perm = idx_b[torch.randperm(idx_b.numel(), device=device)[:n_sample]]
        for a in perm:
            others = idx_b[idx_b != a]
            if same_modality_only:
                others = others[flat_mod[others] == flat_mod[a]]
            if others.numel() < 2:
                continue
            d = (flat_step[others] - flat_step[a]).abs()
            order = torch.argsort(d)
            near, far = others[order[0]], others[order[-1]]
            if d[order[-1]] <= d[order[0]]:
                continue
            s_near = (flat_z[a] * flat_z[near]).sum() / temp
            s_far = (flat_z[a] * flat_z[far]).sum() / temp
            rank_terms.append(F.relu(torch.tensor(margin, device=device) - (s_near - s_far)))
    l_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=device)

    return lambda_multipos * l_multipos + lambda_rank * l_rank, {
        "l_multipos": l_multipos.item(), "l_rank": l_rank.item(),
        "n_pos_anchors": int(has_pos.sum().item()), "n_rank_triplets": len(rank_terms),
    }
