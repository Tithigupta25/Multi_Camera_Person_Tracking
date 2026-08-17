import numpy as np


class SimilarityMatcher:
    """Cosine-similarity utilities for normalized appearance embeddings."""

    def __init__(self, threshold=0.65):
        self.threshold = float(threshold)

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return 0.0

        a = np.asarray(embedding1, dtype=np.float32).reshape(-1)
        b = np.asarray(embedding2, dtype=np.float32).reshape(-1)

        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0

        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def best_match(self, query_embedding, gallery):
        if query_embedding is None or not gallery:
            return None, 0.0

        best_identity = None
        best_similarity = -1.0

        for identity, stored_embedding in gallery.items():
            score = self.cosine_similarity(query_embedding, stored_embedding)
            if score > best_similarity:
                best_identity = identity
                best_similarity = score

        return best_identity, float(best_similarity)

    def is_match(self, embedding1, embedding2):
        score = self.cosine_similarity(embedding1, embedding2)
        return score >= self.threshold, score
