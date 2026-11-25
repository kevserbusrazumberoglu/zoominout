# The spiral traversal engine. One loop performs the four steps img_to_txt,
# txt_shift, txt_to_img and img_shift, and the img_shift step is where the
# surface-mode hard negatives are mined.
# Extracted from the cell HÜCRE 4 of notebooks/ZoomInOut_patch_level.ipynb.
# Note that secrets.choice is not seedable, so a traversal run is not
# reproducible step for step; the generated pairs are deposited at Zenodo for
# that reason.

import secrets
from typing import List, Optional, Set

import numpy as np

from .index import HybridMultimodalIndex, HybridStepRecord


class MultimodalSpiralEngine:
    def __init__(self, index: HybridMultimodalIndex, similarity_threshold: float = 0.55,
                 max_iou_shift: float = 0.1, verbose: bool = True, min_win_size: int = 2,
                 hardness_min: float = 0.05, hard_neg_sem_thresh: float = 0.85,
                 mode_calib: Optional[dict] = None,
                 allowed_image_ids: Optional[Set[str]] = None,
                 top_k: int = 1, diversity_alpha: float = 0.0):
        self.index = index
        self.threshold = similarity_threshold
        self.max_iou_shift = max_iou_shift
        self.verbose = verbose
        self.min_win_size = min_win_size
        self.hardness_min = hardness_min
        self.hard_neg_sem_thresh = hard_neg_sem_thresh
        self._mode_field = {"semantic": "embedding", "color": "color_feat", "shape": "shape_feat", "texture": "texture_feat"}
        # Per-mode calibration samples, used to compare the three modes on a
        # common scale. Leaving it empty makes _calibrated return the raw
        # similarity unchanged.
        self.mode_calib = mode_calib or {}
        # Train/test leakage guard. Without this restriction the txt_to_img
        # step searched the whole index regardless of which split the chain
        # started in, so a training chain could land on a test image. When
        # allowed_image_ids is given the search pool is restricted to it.
        self.allowed_image_ids = set(allowed_image_ids) if allowed_image_ids is not None else None
        if self.allowed_image_ids is not None:
            self._forbidden_image_ids = set(self.index.image_ids()) - self.allowed_image_ids
        else:
            self._forbidden_image_ids = set()
        # Hub-effect correction. query_text and query_image were a plain
        # argmax, so different queries with nearby embeddings converged on the
        # same target, and because nothing was shared across the 600 sequences
        # the convergence repeated silently: only 43% of the training pool was
        # ever visited. The engine now selects among top_k candidates after
        # penalising each by use_count * diversity_alpha. The counters persist
        # across all run_hybrid_loop calls, which makes the correction adaptive
        # over the whole generation run.
        self.top_k = top_k
        self.diversity_alpha = diversity_alpha
        self.image_use_count = {}
        self.doc_use_count = {}

    def _get_feat(self, patch, mode: str):
        return getattr(patch, self._mode_field.get(mode, "embedding"), None)

    def _calibrated(self, raw_sim: float, mode: str) -> float:
        # Percentile-rank normalisation, not a z-score: a z-score inflated the
        # modes whose standard deviation is very small, shape above all.
        # mode_calib[mode] is the sorted sample returned by
        # calibrate_mode_stats, and the rank of raw_sim within it is returned
        # as a value in [0, 1].
        sorted_sims = self.mode_calib.get(mode)
        if sorted_sims is None or not hasattr(sorted_sims, "__len__") or len(sorted_sims) == 0:
            return raw_sim
        rank = np.searchsorted(sorted_sims, raw_sim) / len(sorted_sims)
        return float(rank)

    def _cosine_sim(self, a, b, fallback=0.0):
        if a is None or b is None:
            return fallback
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return fallback
        return float(np.dot(a, b) / (na * nb))

    def _pick_diverse(self, candidates, use_count, key_fn):
        # Selects among the top-K candidates by a score penalised with the
        # candidate's use count, so that hubs fall behind. The use counter of
        # the selected candidate is incremented. The returned score is the raw,
        # unpenalised one, so that logs stay comparable.
        if self.diversity_alpha <= 0.0 or len(candidates) <= 1:
            best = candidates[0]
        else:
            best = max(candidates, key=lambda item: item[1] - self.diversity_alpha * use_count.get(key_fn(item[0]), 0))
        key = key_fn(best[0])
        use_count[key] = use_count.get(key, 0) + 1
        return best

    def run_hybrid_loop(self, start_image_id: Optional[str] = None, loops: int = 16) -> List[HybridStepRecord]:
        log: List[HybridStepRecord] = []
        all_image_ids = self.index.image_ids()
        if self.allowed_image_ids is not None:
            all_image_ids = [i for i in all_image_ids if i in self.allowed_image_ids]
        if not all_image_ids:
            return log

        if start_image_id is None or start_image_id not in all_image_ids:
            start_image_id = secrets.choice(all_image_ids)

        visited_image_ids: Set[str] = {start_image_id}
        visited_doc_ids:   Set[str] = set()
        visited_patch_ids: Set[str] = set()

        pool = self.index.image_patches_of(start_image_id)
        obj_pool = [p for p in pool if p.type == "object"]
        cur_patch = secrets.choice(obj_pool if obj_pool else pool)
        visited_patch_ids.add(cur_patch.patch_id)

        log.append(HybridStepRecord(step=0, action="start", modality="image",
                                     image_id=start_image_id, patch_id=cur_patch.patch_id,
                                     bbox=cur_patch.bbox, similarity_mode="semantic"))
        step_n = 1

        res = self.index.query_text(cur_patch.embedding, exclude_doc_ids=visited_doc_ids,
                                     min_win_size=self.min_win_size, threshold=self.threshold, top_k=self.top_k)
        if not res:
            res = self.index.query_text(cur_patch.embedding, exclude_doc_ids=visited_doc_ids,
                                         min_win_size=self.min_win_size, threshold=-1.0, top_k=self.top_k)
        if not res:
            return log

        tgt_txt, sim0 = self._pick_diverse(res, self.doc_use_count, lambda p: p.doc_id)
        visited_doc_ids.add(tgt_txt.doc_id)
        visited_patch_ids.add(tgt_txt.patch_id)
        hn_txt = self.index.find_hard_negative_text(tgt_txt, visited_doc_ids,
                                                      hardness_min=self.hardness_min,
                                                      hard_neg_sem_thresh=self.hard_neg_sem_thresh)
        log.append(HybridStepRecord(step=step_n, action="img_to_txt", modality="text",
                                     doc_id=tgt_txt.doc_id, patch_id=tgt_txt.patch_id,
                                     text_content=tgt_txt.text_content, similarity=sim0, similarity_mode="semantic",
                                     hard_negative_txt_patch_id=hn_txt[0] if hn_txt else None,
                                     hard_negative_txt_sim=hn_txt[1] if hn_txt else None,
                                     hard_negative_txt_mode=hn_txt[2] if hn_txt else None))
        step_n += 1

        for lp in range(1, loops + 1):
            txt_pool = [p for p in self.index.text_patches_of(tgt_txt.doc_id)
                        if p.patch_id not in visited_patch_ids and p.win_size >= self.min_win_size
                        and self.index._is_quality_text(p.text_content)]
            if not txt_pool:
                txt_pool = [p for p in self.index.text_patches_of(tgt_txt.doc_id) if p.patch_id not in visited_patch_ids]
            if txt_pool:
                tgt_txt = max(txt_pool, key=lambda p: self._cosine_sim(tgt_txt.embedding, p.embedding))
                visited_patch_ids.add(tgt_txt.patch_id)
                log.append(HybridStepRecord(step=step_n, action="txt_shift", modality="text",
                                             doc_id=tgt_txt.doc_id, patch_id=tgt_txt.patch_id,
                                             text_content=tgt_txt.text_content, similarity_mode="semantic"))
                step_n += 1

            # In addition to the visited images, every image outside the
            # allowed pool is excluded, so a training sequence cannot jump to a
            # test image or the other way round.
            img_res = self.index.query_image(tgt_txt.embedding, exclude_image_ids=visited_image_ids | self._forbidden_image_ids, top_k=self.top_k)
            if not img_res:
                break
            best_img, best_score = self._pick_diverse(img_res, self.image_use_count, lambda p: p.image_id)
            visited_image_ids.add(best_img.image_id)
            visited_patch_ids.add(best_img.patch_id)
            log.append(HybridStepRecord(step=step_n, action="txt_to_img", modality="image",
                                         image_id=best_img.image_id, patch_id=best_img.patch_id,
                                         bbox=best_img.bbox, similarity=best_score, similarity_mode="semantic"))
            step_n += 1
            cur_patch = best_img

            MIN_OBJ = 0.4
            img_pool_raw = self.index.image_patches_of(cur_patch.image_id)
            valid_pool = [p for p in img_pool_raw if p.patch_id not in visited_patch_ids
                          and cur_patch.iou(p) <= self.max_iou_shift and p.objectness_score >= MIN_OBJ]
            if not valid_pool:
                valid_pool = sorted([p for p in img_pool_raw if p.patch_id not in visited_patch_ids],
                                     key=lambda x: x.objectness_score, reverse=True)[:5]

            if valid_pool:
                best_mode, best_patch = None, None
                best_surface_sim, best_sem_sim, best_hardness = 0.0, 0.0, -999.0
                for m in ("color", "shape", "texture"):
                    q_feat = self._get_feat(cur_patch, m)
                    if q_feat is None:
                        continue
                    candidate = max(valid_pool, key=lambda p: self._cosine_sim(q_feat, self._get_feat(p, m), fallback=-1.0))
                    surface_sim = self._cosine_sim(q_feat, self._get_feat(candidate, m), fallback=-1.0)
                    sem_sim = self._cosine_sim(cur_patch.embedding, candidate.embedding)
                    # The hardness comparison uses the calibrated surface
                    # similarity, while the reported best_surface_sim stays the
                    # raw value.
                    surface_sim_z = self._calibrated(surface_sim, m)
                    hardness = surface_sim_z - sem_sim
                    if hardness > best_hardness:
                        best_mode, best_patch = m, candidate
                        best_surface_sim, best_sem_sim, best_hardness = surface_sim, sem_sim, hardness

                # Because best_hardness is a difference of calibrated values,
                # hardness_min is not a constant: it has to be recalibrated
                # whenever the descriptor scale changes. See the cell HÜCRE E1
                # for the sweep that selects it.
                if best_patch is not None and best_hardness >= self.hardness_min:
                    mode = best_mode
                    tgt_img = best_patch
                    score = best_surface_sim
                    computed_label = "hard_negative" if best_sem_sim < self.hard_neg_sem_thresh else "positive"
                else:
                    mode = "semantic"
                    tgt_img = secrets.choice(valid_pool)
                    score = self._cosine_sim(cur_patch.embedding, tgt_img.embedding)
                    computed_label = "positive"
            else:
                mode = "semantic"
                score = 0.0
                computed_label = "positive"
                others = [p for p in img_pool_raw if p.patch_id != cur_patch.patch_id]
                tgt_img = min(others, key=lambda p: cur_patch.iou(p)) if others else cur_patch

            visited_patch_ids.add(tgt_img.patch_id)
            log.append(HybridStepRecord(step=step_n, action="img_shift", modality="image",
                                         image_id=tgt_img.image_id, patch_id=tgt_img.patch_id,
                                         bbox=tgt_img.bbox, similarity=score,
                                         similarity_mode=mode, verified_label=computed_label))
            step_n += 1
            cur_patch = tgt_img

            res = self.index.query_text(cur_patch.embedding, exclude_doc_ids=visited_doc_ids,
                                         min_win_size=self.min_win_size, threshold=self.threshold, top_k=self.top_k)
            if not res:
                res = self.index.query_text(cur_patch.embedding, exclude_doc_ids=visited_doc_ids,
                                             min_win_size=self.min_win_size, threshold=-1.0, top_k=self.top_k)
            if not res:
                break
            tgt_txt, sim = self._pick_diverse(res, self.doc_use_count, lambda p: p.doc_id)
            visited_doc_ids.add(tgt_txt.doc_id)
            visited_patch_ids.add(tgt_txt.patch_id)
            hn_txt = self.index.find_hard_negative_text(tgt_txt, visited_doc_ids,
                                                          hardness_min=self.hardness_min,
                                                          hard_neg_sem_thresh=self.hard_neg_sem_thresh)
            log.append(HybridStepRecord(step=step_n, action="img_to_txt", modality="text",
                                         doc_id=tgt_txt.doc_id, patch_id=tgt_txt.patch_id,
                                         text_content=tgt_txt.text_content, similarity=sim, similarity_mode="semantic",
                                         hard_negative_txt_patch_id=hn_txt[0] if hn_txt else None,
                                         hard_negative_txt_sim=hn_txt[1] if hn_txt else None,
                                         hard_negative_txt_mode=hn_txt[2] if hn_txt else None))
            step_n += 1

        return log
