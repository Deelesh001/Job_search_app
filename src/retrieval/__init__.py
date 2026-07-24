from .bm25 import BM25
from .hybrid import reciprocal_rank_fusion

# DISABLED for Cloud Run free tier (512 MB RAM limit):
# DenseRetriever requires fastembed + ONNX Runtime which allocates ~1.5 GB
# at initialization regardless of model size. Re-enable when deploying to
# paid tier (2-4 GB RAM). Implementation remains in src/retrieval/dense.py.
#
# from .dense import DenseRetriever

__all__ = ["BM25", "reciprocal_rank_fusion"]