# Pair dataset, training loop and retrieval evaluation.
# Extracted from the cell HÜCRE 12 of notebooks/ZoomInOut_patch_level.ipynb.
# The notebook built a module-level train_loader from its own train_pairs
# global; here the loader is passed in instead, and nothing else is changed.

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .losses import (MODE_AUX_WEIGHT, MODE_MAP, hn_info_nce, plain_info_nce)
from .models import (InstructionAwareHead, ProjectionHead, SelfInferringHead)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BATCH_SIZE = 128
LR = 1e-4


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = torch.tensor(np.asarray(p["img_embedding"], dtype=np.float32))
        txt = torch.tensor(np.asarray(p["txt_embedding"], dtype=np.float32))
        hn = p.get("hard_negative_embedding")
        has_hn = hn is not None
        hn_t = torch.tensor(np.asarray(hn, dtype=np.float32)) if has_hn else torch.zeros_like(img)
        # Text-side hard negative, present only in the regenerated dataset.
        hn_txt = p.get("hard_negative_txt_embedding")
        has_hn_txt = hn_txt is not None
        hn_txt_t = torch.tensor(np.asarray(hn_txt, dtype=np.float32)) if has_hn_txt else torch.zeros_like(txt)
        mode = p.get("hard_negative_mode") or "semantic"
        mode_idx = MODE_MAP.get(mode, 0)
        # Optional COCO-category hard negative. When it is absent has_chn is 0
        # and the other arms are unaffected.
        chn = p.get("category_hard_negative_embedding")
        has_chn = chn is not None
        chn_t = torch.tensor(np.asarray(chn, dtype=np.float32)) if has_chn else torch.zeros_like(img)
        return {"img": img, "txt": txt, "hn": hn_t,
                "has_hn": torch.tensor(1.0 if has_hn else 0.0),
                "hn_txt": hn_txt_t,
                "has_hn_txt": torch.tensor(1.0 if has_hn_txt else 0.0),
                "chn": chn_t,
                "has_chn": torch.tensor(1.0 if has_chn else 0.0),
                "mode_idx": torch.tensor(mode_idx, dtype=torch.long)}


def make_loader(pairs, batch_size=BATCH_SIZE):
    return DataLoader(PairDataset(pairs), batch_size=batch_size,
                      shuffle=True, drop_last=True)


def train_model(model_type, loss_fn_name, epochs, loader, val_pairs=None,
                patience=4, eval_every=5, seed=None):
    # Cosine learning-rate schedule, validation-based early stopping and
    # best-checkpoint restoration. When val_pairs is given, validation R@1 is
    # measured every eval_every epochs, the weights of the best check are kept
    # and restored at the end, so the returned model is the best epoch by
    # validation rather than the last one. Training stops early if no check
    # improves for patience rounds.
    # The seed parameter matters: the loop previously reset to the global SEED
    # constant on every call, which silently erased the seed variation of the
    # multi-seed validation and made every arm return identical numbers across
    # seeds.
    torch.manual_seed(seed if seed is not None else SEED)
    if model_type == "plain":
        model = ProjectionHead().to(device)
    elif model_type == "instruction":
        model = InstructionAwareHead().to(device)
    else:
        model = SelfInferringHead().to(device)

    loss_fn = plain_info_nce if loss_fn_name == "plain" else hn_info_nce
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_val_r1 = -1.0
    checks_since_improve = 0

    data_loader = loader

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_aux = 0.0, 0.0
        for batch in data_loader:
            img = batch["img"].to(device); txt = batch["txt"].to(device)
            hn = batch["hn"].to(device); has_hn = batch["has_hn"].to(device)
            mode_idx = batch["mode_idx"].to(device)

            optimizer.zero_grad()
            if model_type == "plain":
                z_img, z_txt = model(img, txt)
                z_hn = model.project_image(hn) if loss_fn_name == "hn" else None
                loss = loss_fn(z_img, z_txt, z_hn=z_hn, has_hn=has_hn)
                aux = torch.tensor(0.0)
            elif model_type == "instruction":
                z_img, z_txt = model(img, txt, mode_idx)
                z_hn = model.project_image(hn, mode_idx) if loss_fn_name == "hn" else None
                loss = loss_fn(z_img, z_txt, z_hn=z_hn, has_hn=has_hn)
                aux = torch.tensor(0.0)
            else:
                z_img, z_txt, mode_logits = model(img, txt)
                z_hn = None
                if loss_fn_name == "hn":
                    mode_vec_hn, _ = model.infer_mode(img, txt)
                    z_hn = model.project_image_with_vec(hn, mode_vec_hn)
                base_loss = loss_fn(z_img, z_txt, z_hn=z_hn, has_hn=has_hn)
                mask = has_hn.bool()
                aux = F.cross_entropy(mode_logits[mask], mode_idx[mask]) if mask.sum() > 0 else torch.tensor(0.0)
                loss = base_loss + MODE_AUX_WEIGHT * aux

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); epoch_aux += aux.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_type}/{loss_fn_name}] epoch {epoch+1}/{epochs} loss={epoch_loss/len(data_loader):.4f} mode_aux={epoch_aux/len(data_loader):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if val_pairs is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
            r_val = evaluate(model, model_type, val_pairs, forced_mode=None)
            if r_val["R@1"] > best_val_r1:
                best_val_r1 = r_val["R@1"]
                best_state = copy.deepcopy(model.state_dict())
                checks_since_improve = 0
            else:
                checks_since_improve += 1
            if checks_since_improve >= patience:
                print(f"  [early stop] val R@1 did not improve for {patience} checks (best {best_val_r1:.1f}%), stopping at epoch {epoch+1}.")
                break

    if val_pairs is not None:
        model.load_state_dict(best_state)
        print(f"  [{model_type}/{loss_fn_name}] restored the checkpoint with the best val R@1={best_val_r1:.1f}%.")
    return model


@torch.no_grad()
def evaluate(model, model_type, pairs, forced_mode=None):
    model.eval()
    img_all = torch.tensor(np.stack([np.asarray(p["img_embedding"], dtype=np.float32) for p in pairs])).to(device)
    txt_all = torch.tensor(np.stack([np.asarray(p["txt_embedding"], dtype=np.float32) for p in pairs])).to(device)

    if model_type == "plain":
        z_img, z_txt = model(img_all, txt_all)
    elif model_type == "instruction":
        if forced_mode is not None:
            mode_idx = torch.full((len(pairs),), MODE_MAP[forced_mode], dtype=torch.long, device=device)
        else:
            mode_idx = torch.tensor([MODE_MAP.get(p.get("hard_negative_mode") or "semantic", 0) for p in pairs], dtype=torch.long, device=device)
        z_img, z_txt = model(img_all, txt_all, mode_idx)
    else:
        z_img, z_txt, mode_logits = model(img_all, txt_all)
        pred_mode = mode_logits.argmax(dim=-1)
        true_mode = torch.tensor([MODE_MAP.get(p.get("hard_negative_mode") or "semantic", 0) for p in pairs], dtype=torch.long, device=device)
        acc = (pred_mode == true_mode).float().mean().item()
        print(f"  [self_infer] mode prediction accuracy, reported for information only and not used as an oracle: {acc*100:.1f}%")

    sim = (z_txt @ z_img.T).cpu().numpy()
    n = len(pairs)
    r1 = r5 = r10 = 0
    mrr = 0.0
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        if rank == 1: r1 += 1
        if rank <= 5: r5 += 1
        if rank <= 10: r10 += 1
        mrr += 1.0 / rank
    return {"R@1": r1/n*100, "R@5": r5/n*100, "R@10": r10/n*100, "MRR": mrr/n}
