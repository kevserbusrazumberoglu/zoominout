# ZoomInOut: spiral-traversal hard negative generation for cross-modal
# contrastive learning.
# The modules here are extracted from the notebooks under notebooks/ so that
# the method can be read without running a notebook. The reported numbers were
# produced by the notebooks themselves.

from .index import (HybridMultimodalIndex, HybridStepRecord, ImagePatchMeta,
                    TextPatchMeta)
from .engine import MultimodalSpiralEngine
from .descriptors import (attach_surface_descriptors, calibrate_mode_stats,
                          compute_surface_descriptors)
from .generation import (attach_embeddings, build_pairs_from_sequences,
                         carve_val_split, generate_sequences,
                         generate_test_pairs, make_image_split)

__all__ = [
    "HybridMultimodalIndex", "HybridStepRecord", "ImagePatchMeta",
    "TextPatchMeta", "MultimodalSpiralEngine", "attach_surface_descriptors",
    "calibrate_mode_stats", "compute_surface_descriptors",
    "attach_embeddings", "build_pairs_from_sequences", "carve_val_split",
    "generate_sequences", "generate_test_pairs", "make_image_split",
]
