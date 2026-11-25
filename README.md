# ZoomInOut

Code for the paper *ZoomInOut: Spiral-Traversal-Based Hard Negative Generation
for Cross-Modal Contrastive Learning*.

The method mines hard negatives for cross-modal contrastive learning by a
spiral traversal that alternates between image patches and sentence windows.
Each mined negative is labelled with the surface property, colour, shape or
texture, that makes it confusable with its anchor, and a mode-conditioned
residual adapter is trained on top of a frozen backbone using that label.

## What is in this repository

```
zoominout/          the method, extracted from the notebooks
  index.py          data classes and the hybrid image/text index
  descriptors.py    colour, shape and texture descriptors, mode calibration
  engine.py         the spiral traversal engine and hard negative mining
  generation.py     split logic, sequence generation, pair construction
  models.py         residual adapter and its mode-conditioned variants
  losses.py         InfoNCE, HN-InfoNCE and the auxiliary mode term
  training.py       pair dataset, training loop, retrieval evaluation
  chain_loss.py     chain-distance loss, used only in one ablation
notebooks/          the notebooks that produced the reported numbers
  ZoomInOut_patch_level.ipynb   patch-level traversal, the main experiment
  ZoomInOut_whole_level.ipynb   the same algorithm over whole images
  ZoomInOut_pool_scale.ipynb    the 10K pool experiment
```

The modules under `zoominout/` are extracted from the notebook cells so that
the method can be read and reused without opening a notebook. Each module names
the cell it came from. The numbers in the paper were produced by the notebooks
themselves, which is why both are kept here.

## Which cell produced which table

| Paper | Notebook | Cell |
| --- | --- | --- |
| Table 1, dataset summary | `ZoomInOut_patch_level.ipynb` | `HÜCRE L2` |
| Table 2, pool coverage before and after the hub correction | `ZoomInOut_patch_level.ipynb` | `HÜCRE L2`, together with the diversity settings of `HÜCRE 6` |
| Algorithm 1 | `ZoomInOut_patch_level.ipynb` | `HÜCRE 4`, `HÜCRE 6` |
| Table 3, main results, patch-level block | `ZoomInOut_patch_level.ipynb` | `HÜCRE L4-TIMED`, `HÜCRE R-FIX`, `HÜCRE L5` |
| Table 3, main results, whole-image block | `ZoomInOut_whole_level.ipynb` | `HÜCRE W6`, evaluated on the test pairs of `HÜCRE 4` |
| Table 4, chain distance as an explicit loss term | `ZoomInOut_patch_level.ipynb` | `HÜCRE L6`, `HÜCRE L6c` |
| Table 5, data pool scale | `ZoomInOut_pool_scale.ipynb` | `HÜCRE SEQ2`, `HÜCRE SEQ3` |
| Threshold calibration | `ZoomInOut_patch_level.ipynb` | `HÜCRE E1`, `HÜCRE E2` |
| Threshold calibration, whole-image | `ZoomInOut_whole_level.ipynb` | `HÜCRE W4c`, `HÜCRE W4d` |

The cell titles appear as the first comment line of each cell, so they can be
found by searching the notebook for the cell name.

## Data

The generated training, validation and test pairs, the image identifiers of
the 2K and 10K pools, the split assignment, the hard negative annotations with
their surface mode labels and the mode calibration thresholds are deposited at
Zenodo:

https://doi.org/10.5281/zenodo.24527757

Neither the images nor the corpus text are redistributed there. The records
refer to MS COCO images by their COCO identifier and to the narrative corpus by
window identifier. The images come from MS COCO and the narrative corpus is a
Turkish translation of SimpleStories, which is distributed under the MIT
licence.

The embeddings are not deposited, because they are specific to the frozen
backbone. To retrain, the pairs have to be re-embedded with the backbone of
choice; the notebooks show how the embedding tables are built and cached.

## Reproducing

The experiments were run in Google Colab on a single GPU, with the datasets and
the embedding caches kept in Google Drive under `MyDrive/COSMOS/ZoomInOut`. The
notebooks expect that layout and each caching cell checks for its own pickle
before recomputing, so a run can be interrupted and resumed.

Two things are worth knowing before a rerun.

The traversal is not reproducible step for step. The engine selects the
starting patch and some fallback targets with `secrets.choice`, which cannot be
seeded, so a new traversal produces different sequences. The pair statistics
are stable but the individual chains are not. The deposited pairs are the ones
the reported numbers were computed from.

The hardness threshold is calibrated, not fixed. `hardness_min` is a difference
of percentile ranks, so it depends on the scale of the descriptors and of the
backbone similarities. It has to be reswept whenever either changes. The cell
`HÜCRE E1` performs that sweep at patch level and `HÜCRE W4c` its counterpart
over whole images; the whole-image descriptors have a much narrower similarity
distribution and need a clearly different threshold.

Training itself is cheap. The backbone stays frozen and only the adapter is
updated, so one arm takes well under a minute on one GPU, while generating the
data takes considerably longer.

## Installation

```
pip install -r requirements.txt
```

The modules under `zoominout/` need only NumPy, OpenCV, scikit-image, tqdm and
PyTorch. The notebooks additionally load the backbone and the datasets, and
their first cell installs what they need.

## Citation

See `CITATION.cff`.

## Licence

The code in this repository is released under the MIT licence, see `LICENSE`.
The data it refers to is licensed separately, see `NOTICE`.
