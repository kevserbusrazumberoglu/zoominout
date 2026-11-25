# Generation settings, the image-level split and the pair construction
# functions.
# Extracted from the cell HÜCRE 6 of notebooks/ZoomInOut_patch_level.ipynb.
# attach_embeddings read the COCO caption tables from notebook globals; here
# they are passed in as an optional argument, and nothing else is changed.

import random

import numpy as np
from tqdm.auto import tqdm

NUM_SEQUENCES        = 600
LOOPS_PER_SEQUENCE   = 16
MIN_CHAIN_LENGTH     = 16
SIM_THRESHOLD        = 0.55
TRAIN_RATIO          = 0.75

TEST_NUM_SEQUENCES   = 600
# 400 unique source images is a reachable target in a 500-image test pool.
TEST_TARGET_SAMPLES  = 400

# 15% of the training pool is held out for early stopping and model selection.
VAL_RATIO            = 0.15
VAL_NUM_SEQUENCES    = 300
VAL_TARGET_SAMPLES   = 150

TOP_K              = 8
DIVERSITY_ALPHA    = 0.03


def make_image_split(hybrid_index, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    all_ids = hybrid_index.image_ids()
    rng.shuffle(all_ids)
    split_point = int(len(all_ids) * train_ratio)
    return set(all_ids[:split_point]), set(all_ids[split_point:])


def carve_val_split(train_image_ids, val_ratio=0.15, seed=43):
    # Holds out part of the training pool for early stopping and model
    # selection. The test pool is never touched. sorted() is applied first so
    # that the result does not depend on set iteration order.
    rng = random.Random(seed)
    all_ids = sorted(train_image_ids)
    rng.shuffle(all_ids)
    split_point = int(len(all_ids) * (1 - val_ratio))
    return set(all_ids[:split_point]), set(all_ids[split_point:])


def generate_sequences(hybrid_index, engine, image_ids, num_sequences, loops, min_chain_length):
    # Coverage-guaranteed selection of the starting images. Each draw used to
    # be independently random, so over 600 draws only about 38% of the
    # 1275-image pool was ever used as a start, which is the coupon-collector
    # expectation. The pool is now shuffled once and traversed in order, so in
    # the first min(num_sequences, len(image_ids)) draws every image starts
    # exactly one sequence, and the traversal wraps around beyond that.
    image_ids = list(image_ids)
    _rng_starts = random.Random(1234)
    shuffled = image_ids[:]
    _rng_starts.shuffle(shuffled)
    sequences = []
    for i in tqdm(range(num_sequences), desc="generating sequences"):
        start_img = shuffled[i % len(shuffled)]
        chain = engine.run_hybrid_loop(start_image_id=start_img, loops=loops)
        if len(chain) >= min_chain_length:
            sequences.append(chain)
    print(f"{len(sequences)}/{num_sequences} sequences kept "
          f"(passed the min_chain_length={min_chain_length} filter)")
    return sequences


def build_pairs_from_sequences(sequences, sim_threshold):
    # Cross-modal positive pairs, with a hard negative attached only from the
    # img_shift steps whose verified_label is "hard_negative".
    records = []
    n_with_hn = 0
    for chain in sequences:
        for i, step in enumerate(chain):
            if step.action not in ("img_to_txt", "txt_to_img") or i == 0:
                continue
            prev = chain[i - 1]
            img_step, txt_step = (prev, step) if step.action == "img_to_txt" else (step, prev)
            if img_step.modality != "image" or txt_step.modality != "text":
                continue
            cross_sim = step.similarity if step.similarity else 0.0
            if cross_sim < sim_threshold:
                continue

            record = {
                "txt_patch_id": txt_step.patch_id, "img_patch_id": img_step.patch_id,
                "cross_modal_sim": cross_sim,
                "hard_negative_patch_id": None, "hard_negative_sim": None, "hard_negative_mode": None,
                # Text-side hard negative. It is filled only when txt_step is
                # an img_to_txt step; in a txt_to_img pair txt_step is a
                # txt_shift and these fields stay None, which gives a coverage
                # of roughly 50%.
                "hard_negative_txt_patch_id": getattr(txt_step, "hard_negative_txt_patch_id", None),
                "hard_negative_txt_sim": getattr(txt_step, "hard_negative_txt_sim", None),
                "hard_negative_txt_mode": getattr(txt_step, "hard_negative_txt_mode", None),
            }
            if i + 1 < len(chain) and chain[i + 1].action == "img_shift":
                shift_step = chain[i + 1]
                if (shift_step.image_id == img_step.image_id
                        and getattr(shift_step, "verified_label", "positive") == "hard_negative"):
                    record["hard_negative_patch_id"] = shift_step.patch_id
                    record["hard_negative_sim"] = shift_step.similarity
                    record["hard_negative_mode"] = shift_step.similarity_mode
                    n_with_hn += 1
            records.append(record)

    print(f"cross-modal pairs: {len(records)} | with a verified hard negative: "
          f"{n_with_hn} ({n_with_hn/max(len(records),1)*100:.1f}%)")
    return records


def attach_embeddings(records, hybrid_index, coco_captions=None,
                      patch_image_id_to_coco_id=None):
    txt_lookup = {p.patch_id: p.embedding for p in hybrid_index.text_patches}
    img_lookup = {p.patch_id: p.embedding for p in hybrid_index.image_patches}
    # Needed to map a patch back to its parent image for the caption lookup.
    img_id_lookup = {p.patch_id: p.image_id for p in hybrid_index.image_patches}
    coco_mapping_ready = (coco_captions is not None
                          and patch_image_id_to_coco_id is not None)
    final = []
    n_with_hn_txt = 0
    n_coco_caption_used = 0
    for r in records:
        txt_emb = txt_lookup.get(r["txt_patch_id"])
        img_emb = img_lookup.get(r["img_patch_id"])
        if txt_emb is None or img_emb is None:
            continue

        # When a COCO caption is available, the positive text is the image's
        # own caption rather than the nearest story sentence. The story windows
        # visited during traversal therefore serve only as bridges.
        if coco_mapping_ready:
            img_image_id = img_id_lookup.get(r["img_patch_id"])
            coco_id = patch_image_id_to_coco_id.get(img_image_id)
            if coco_id is not None and coco_id in coco_captions:
                _cap_text, _cap_emb = random.choice(coco_captions[coco_id])
                txt_emb = _cap_emb
                n_coco_caption_used += 1

        hn_emb = img_lookup.get(r["hard_negative_patch_id"]) if r["hard_negative_patch_id"] is not None else None
        hn_txt_emb = txt_lookup.get(r["hard_negative_txt_patch_id"]) if r.get("hard_negative_txt_patch_id") is not None else None
        if hn_txt_emb is not None:
            n_with_hn_txt += 1
        final.append({
            "txt_embedding": np.asarray(txt_emb, dtype=np.float32),
            "img_embedding": np.asarray(img_emb, dtype=np.float32),
            "cross_modal_sim": r["cross_modal_sim"],
            "hard_negative_embedding": np.asarray(hn_emb, dtype=np.float32) if hn_emb is not None else None,
            "hard_negative_sim": r["hard_negative_sim"],
            "hard_negative_mode": r["hard_negative_mode"],
            "hard_negative_txt_embedding": np.asarray(hn_txt_emb, dtype=np.float32) if hn_txt_emb is not None else None,
            "hard_negative_txt_sim": r.get("hard_negative_txt_sim"),
        })
    print(f"records with complete embeddings: {len(final)} "
          f"(with a text-side hard negative: {n_with_hn_txt}, "
          f"{n_with_hn_txt/max(len(final),1)*100:.1f}%)")
    if coco_mapping_ready:
        print(f"positive texts replaced by a COCO caption: {n_coco_caption_used} "
              f"({n_coco_caption_used/max(len(final),1)*100:.1f}%)")
    return final


def generate_test_pairs(hybrid_index, engine, test_image_ids, num_sequences, loops,
                         min_chain_length, sim_threshold, target_samples,
                         coco_captions=None, patch_image_id_to_coco_id=None):
    # Two defects were corrected here. First, the same image could appear in
    # the test set many times, which inflated R@1; at most one sample per image
    # is kept now. Second, only the records carrying a verified hard negative
    # were retained, which forced the test set into a different distribution
    # from training, 100% against roughly 17%, and narrowed the pool from
    # 17896 records to 2928 and to only 117 unique images. The test set is now
    # sampled from the same unfiltered distribution as training, so the pool is
    # far wider and the unique-image target is reachable.
    test_sequences = []
    test_image_ids = list(test_image_ids)
    _rng_starts = random.Random(1234)
    shuffled = test_image_ids[:]
    _rng_starts.shuffle(shuffled)
    for i in tqdm(range(num_sequences), desc="test sequences"):
        start_img = shuffled[i % len(shuffled)]
        chain = engine.run_hybrid_loop(start_image_id=start_img, loops=loops)
        if len(chain) >= min_chain_length:
            test_sequences.append(chain)

    raw_records = build_pairs_from_sequences(test_sequences, sim_threshold)
    random.shuffle(raw_records)

    img_id_lookup = {p.patch_id: p.image_id for p in hybrid_index.image_patches}
    seen_images = set()
    deduped_records = []
    for r in raw_records:
        src_img_id = img_id_lookup.get(r["img_patch_id"])
        if src_img_id in seen_images:
            continue
        seen_images.add(src_img_id)
        deduped_records.append(r)
        if len(deduped_records) >= target_samples:
            break

    if len(deduped_records) < target_samples:
        print(f"WARNING: the target was {target_samples} but pairs were found "
              f"from only {len(deduped_records)} unique images. The test pool "
              f"(test_image_ids ratio or num_sequences) may need to be larger.")
    print(f"unique source images: {len(deduped_records)} / target {target_samples}")

    final_pairs = attach_embeddings(deduped_records, hybrid_index,
                                    coco_captions=coco_captions,
                                    patch_image_id_to_coco_id=patch_image_id_to_coco_id)
    mode_counts = {}
    for p in final_pairs:
        m = p["hard_negative_mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1
    print("mode distribution:", mode_counts)
    return final_pairs

