# Surface descriptors (colour, shape, texture) computed on raw pixels, and the
# percentile calibration of the per-mode similarity distributions.
# The descriptor body is taken from the cell HÜCRE 5b and calibrate_mode_stats
# from the cell that follows it in notebooks/ZoomInOut_patch_level.ipynb; the
# only change is that the loop body is wrapped in a function so that it can be
# called for a single patch.

import os
import pickle
import random

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm.auto import tqdm


def compute_surface_descriptors(pil_img, bbox):
    # bbox is given in the 224x224 frame the patch proposals were produced in,
    # so it is rescaled to the original image resolution before cropping.
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    H_img, W_img = img_bgr.shape[:2]
    x1 = int(np.clip(bbox[0] * (W_img / 224.0), 0, W_img - 1))
    y1 = int(np.clip(bbox[1] * (H_img / 224.0), 0, H_img - 1))
    x2 = int(np.clip(bbox[2] * (W_img / 224.0), 1, W_img))
    y2 = int(np.clip(bbox[3] * (H_img / 224.0), 1, H_img))
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    crop = img_bgr[y1:y2, x1:x2]

    # colour: HSV histogram, 16 hue + 8 saturation + 8 value bins
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    color_f = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    color_f /= (np.linalg.norm(color_f) + 1e-10)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # texture: 32-bin Sobel gradient magnitude histogram plus a 10-bin
    # uniform LBP histogram
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    hist_txt, _ = np.histogram(mag, bins=32, range=(0, 255))
    hist_txt = hist_txt.astype(np.float32)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
    hist_lbp = hist_lbp.astype(np.float32)

    texture_f = np.concatenate([hist_txt, hist_lbp])
    texture_f /= (np.linalg.norm(texture_f) + 1e-10)

    # shape: 32-bin Canny edge histogram, 7 log-scaled Hu moments and the
    # aspect ratio of the crop
    edges = cv2.Canny(gray, 50, 150)
    hist_edge, _ = np.histogram(edges, bins=32, range=(0, 255))
    moments = cv2.moments(edges)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    hu_log = np.nan_to_num(hu_log, nan=0.0, posinf=0.0, neginf=0.0)
    aspect_ratio = float(x2 - x1) / float(y2 - y1)
    shape_f = np.concatenate([hist_edge.astype(np.float32),
                              hu_log.astype(np.float32),
                              [aspect_ratio]])
    shape_f /= (np.linalg.norm(shape_f) + 1e-10)

    return color_f, shape_f, texture_f


def attach_surface_descriptors(index, visual_loader, cache_path=None):
    # Fills color_feat / shape_feat / texture_feat on every image patch of the
    # index. If a patch cannot be processed its semantic embedding is used as a
    # fallback, which makes the corresponding mode uninformative for it rather
    # than dropping the patch.
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cv_cache = pickle.load(f)
        n_hit = 0
        for patch in index.image_patches:
            feats = cv_cache.get(patch.patch_id)
            if feats is not None:
                patch.color_feat, patch.shape_feat, patch.texture_feat = feats
                n_hit += 1
            else:
                patch.color_feat = patch.embedding
                patch.shape_feat = patch.embedding
                patch.texture_feat = patch.embedding
        print(f"{n_hit}/{len(index.image_patches)} patches filled from cache.")
        return

    for patch in tqdm(index.image_patches, desc="surface descriptors"):
        try:
            pil_img = visual_loader(patch.image_id)
            patch.color_feat, patch.shape_feat, patch.texture_feat = \
                compute_surface_descriptors(pil_img, patch.bbox)
        except Exception:
            patch.color_feat = patch.embedding
            patch.shape_feat = patch.embedding
            patch.texture_feat = patch.embedding

    if cache_path is not None:
        cv_cache = {p.patch_id: (p.color_feat, p.shape_feat, p.texture_feat)
                    for p in index.image_patches}
        with open(cache_path, "wb") as f:
            pickle.dump(cv_cache, f)


def calibrate_mode_stats(index, n_samples=3000, seed=42):
    # Returns, for each mode, the sorted sample of surface similarities between
    # two random patches of the same image. The engine converts a raw surface
    # similarity into a percentile rank against this sample.
    # A z-score was tried first and rejected: modes with a very small standard
    # deviation (shape: 0.0093) produced enormous z-scores for tiny deviations
    # and won the mode competition every time, so colour was selected in 100%
    # of the steps and shape and texture never. The percentile rank is
    # invariant to the shape and the variance of each mode's own distribution.
    rng = random.Random(seed)
    feat_attr = {"color": "color_feat", "shape": "shape_feat", "texture": "texture_feat"}
    image_ids = index.image_ids()
    stats = {}
    for mode, attr in feat_attr.items():
        sims = []
        tries = 0
        while len(sims) < n_samples and tries < n_samples * 5:
            tries += 1
            img_id = rng.choice(image_ids)
            patches = index.image_patches_of(img_id)
            if len(patches) < 2:
                continue
            a, b = rng.sample(patches, 2)
            fa, fb = getattr(a, attr, None), getattr(b, attr, None)
            if fa is None or fb is None:
                continue
            na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
            if na == 0 or nb == 0:
                continue
            sims.append(float(np.dot(fa, fb) / (na * nb)))
        stats[mode] = np.sort(np.asarray(sims, dtype=np.float32))
    return stats
