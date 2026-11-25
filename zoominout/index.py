# Data classes and the hybrid image/text index used by the traversal engine.
# Extracted from the cells HÜCRE 2 and HÜCRE 3 of
# notebooks/ZoomInOut_patch_level.ipynb.

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ImagePatchMeta:
    image_id:         str
    patch_id:         str
    bbox:             list
    embedding:        np.ndarray
    objectness_score: float = 1.0
    type:             str   = "object"
    window_type:      str   = ""
    area:             float = 0.0
    attention_score:  float = 0.0
    color_feat:       Optional[np.ndarray] = None
    shape_feat:       Optional[np.ndarray] = None
    texture_feat:     Optional[np.ndarray] = None

    def iou(self, other: "ImagePatchMeta") -> float:
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        a2 = (other.bbox[2] - other.bbox[0]) * (other.bbox[3] - other.bbox[1])
        return inter / (a1 + a2 - inter + 1e-10)


@dataclass
class TextPatchMeta:
    doc_id:         str
    patch_id:       str
    text_content:   str
    embedding:      np.ndarray
    win_size:       int   = 2
    win_label:      str   = "w2"
    richness_score: float = 0.0
    sent_range:     tuple = field(default_factory=lambda: (0, 0))
    lexical_feat:   Optional[np.ndarray] = None  # TF-IDF -- metin-tarafı hard-negative madenciliği için


@dataclass
class HybridStepRecord:
    step:            int
    action:          str
    modality:        str
    image_id:        Optional[str]   = None
    patch_id:        Optional[str]   = None
    bbox:            Optional[list]  = None
    doc_id:          Optional[str]   = None
    text_content:    Optional[str]   = None
    similarity:      float           = 0.0
    similarity_mode: str             = "semantic"
    verified_label:  str             = "positive"
    # Metin-tarafı hard-negative (sadece img_to_txt/txt_to_img adımlarında dolu olur):
    hard_negative_txt_patch_id: Optional[str]   = None
    hard_negative_txt_sim:      Optional[float] = None
    hard_negative_txt_mode:     Optional[str]   = None



class HybridMultimodalIndex:
    def __init__(self):
        self.image_patches: List[ImagePatchMeta] = []
        self.text_patches:  List[TextPatchMeta]  = []
        self._img_matrix:   Optional[np.ndarray] = None
        self._txt_matrix:   Optional[np.ndarray] = None
        self._lex_matrix:   Optional[np.ndarray] = None
        self._img_dirty = True
        self._txt_dirty = True
        self._lex_dirty = True

    def _is_quality_text(self, text: str) -> bool:
        if not text or len(text.split()) < 5:
            return False
        if text.strip().endswith("?"):
            return False
        BAGLAC = {"ve", "ama", "fakat", "lakin", "çünkü", "ancak",
                  "ile", "ya", "da", "de", "ki", "bu", "şu", "o",
                  "diye", "hem", "ne", "ah", "oh", "hmm"}
        ilk = text.split()[0].lower().rstrip(".,!?")
        if ilk in BAGLAC:
            return False
        if text.startswith('"') or text.startswith('-') or '—' in text:
            return False
        return True

    def add_image_patch(self, patch: ImagePatchMeta):
        for attr in ("embedding", "color_feat", "shape_feat", "texture_feat"):
            feat = getattr(patch, attr, None)
            if feat is not None:
                n = np.linalg.norm(feat)
                if n > 0:
                    setattr(patch, attr, feat / n)
        self.image_patches.append(patch)
        self._img_dirty = True

    def add_text_patch(self, patch: TextPatchMeta):
        n = np.linalg.norm(patch.embedding)
        if n > 0:
            patch.embedding = patch.embedding / n
        if patch.lexical_feat is not None:
            ln = np.linalg.norm(patch.lexical_feat)
            if ln > 0:
                patch.lexical_feat = patch.lexical_feat / ln
        self.text_patches.append(patch)
        self._txt_dirty = True
        self._lex_dirty = True

    def _build_matrices(self):
        if self._img_dirty and self.image_patches:
            self._img_matrix = np.stack([p.embedding for p in self.image_patches], axis=0).astype(np.float32)
            self._img_dirty = False
        if self._txt_dirty and self.text_patches:
            self._txt_matrix = np.stack([p.embedding for p in self.text_patches], axis=0).astype(np.float32)
            self._txt_dirty = False
        if self._lex_dirty and self.text_patches and all(p.lexical_feat is not None for p in self.text_patches):
            self._lex_matrix = np.stack([p.lexical_feat for p in self.text_patches], axis=0).astype(np.float32)
            self._lex_dirty = False

    def query_text(self, query_emb, exclude_doc_ids=None, threshold=-1.0, min_win_size=1, top_k=1):
        # With top_k > 1 the best top_k candidates are returned, ordered by
        # the weighted score, so that the engine can make a
        # diversity-aware choice among them instead of taking a plain argmax.
        self._build_matrices()
        if self._txt_matrix is None or not self.text_patches:
            return []
        exclude_doc_ids = exclude_doc_ids or set()
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        raw = np.clip(self._txt_matrix @ q, -1.0, 1.0)
        WIN_WEIGHT = {1: 0.70, 2: 1.05, 3: 1.15}
        weighted = raw.copy()
        for i, p in enumerate(self.text_patches):
            if p.doc_id in exclude_doc_ids or p.win_size < min_win_size or not self._is_quality_text(p.text_content):
                weighted[i] = -999.0
            else:
                weighted[i] = weighted[i] * WIN_WEIGHT.get(p.win_size, 1.0)
        if top_k <= 1:
            best_idx = int(np.argmax(weighted))
            if weighted[best_idx] <= -999.0:
                return []
            return [(self.text_patches[best_idx], float(raw[best_idx]))]
        order = np.argsort(-weighted)[:top_k]
        return [(self.text_patches[i], float(raw[i])) for i in order if weighted[i] > -999.0]

    def query_image(self, query_emb, exclude_image_ids=None, top_k=1):
        # With top_k > 1 the best top_k candidates are returned instead of
        # only the single best one.
        self._build_matrices()
        if self._img_matrix is None or not self.image_patches:
            return []
        exclude_image_ids = exclude_image_ids or set()
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        raw = np.clip(self._img_matrix @ q, -1.0, 1.0)
        TYPE_WEIGHT = {"object": 1.10, "background": 0.92}
        weighted = raw.copy()
        for i, p in enumerate(self.image_patches):
            if p.image_id in exclude_image_ids:
                weighted[i] = -999.0
            else:
                weighted[i] = weighted[i] * TYPE_WEIGHT.get(p.type, 1.0)
        if top_k <= 1:
            best_idx = int(np.argmax(weighted))
            if weighted[best_idx] <= -999.0:
                return []
            return [(self.image_patches[best_idx], float(raw[best_idx]))]
        order = np.argsort(-weighted)[:top_k]
        return [(self.image_patches[i], float(raw[i])) for i in order if weighted[i] > -999.0]

    def text_patches_of(self, doc_id):
        return [p for p in self.text_patches if p.doc_id == doc_id]

    def image_patches_of(self, image_id):
        return [p for p in self.image_patches if p.image_id == image_id]

    def image_ids(self):
        # Sorted on purpose: set iteration order can differ between Python
        # processes because of hash randomisation, which made the train/test
        # split differ from session to session and leaked across the split.
        return sorted({p.image_id for p in self.image_patches})

    def find_hard_negative_text(self, anchor_patch: "TextPatchMeta", exclude_doc_ids,
                                 hardness_min: float = 0.05, hard_neg_sem_thresh: float = 0.85):
        # Text-side hard negative search: a candidate from a different
        # document that is similar to anchor_patch on the surface, by TF-IDF
        # lexical overlap, but semantically different. This is the text
        # counterpart of the colour, shape and texture logic on the image side.
        # It does not affect the traversal: it is recorded as a parallel field
        # on the img_to_txt step and leaves the semantically-nearest selection
        # of txt_shift, which the chain consistency depends on, untouched.
        self._build_matrices()
        if self._lex_matrix is None or anchor_patch.lexical_feat is None:
            return None
        surface_all = np.clip(self._lex_matrix @ anchor_patch.lexical_feat, -1.0, 1.0)
        sem_all = np.clip(self._txt_matrix @ anchor_patch.embedding, -1.0, 1.0)
        hardness_all = surface_all - sem_all

        best_idx, best_hardness = -1, -999.0
        for i, p in enumerate(self.text_patches):
            if p.doc_id in exclude_doc_ids or p.doc_id == anchor_patch.doc_id:
                continue
            if not self._is_quality_text(p.text_content):
                continue
            if hardness_all[i] > best_hardness:
                best_idx, best_hardness = i, hardness_all[i]

        if best_idx == -1 or best_hardness < hardness_min or sem_all[best_idx] >= hard_neg_sem_thresh:
            return None
        cand = self.text_patches[best_idx]
        return cand.patch_id, float(surface_all[best_idx]), "lexical"
