from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

class BGEDense:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_path, device=device)

    def encode_batch(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return _l2_normalize(embs)