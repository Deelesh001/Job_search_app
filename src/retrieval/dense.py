import numpy as np
from fastembed import TextEmbedding


class DenseRetriever:
    """Dense vector retrieval using fastembed (ONNX Runtime) instead of
    sentence-transformers/torch -- same interface as before (fit_corpus,
    get_scores, get_ranking) so nothing else in the pipeline needs to change.

    NOTE: model_name switched from 'all-MiniLM-L6-v2' to fastembed's
    'BAAI/bge-small-en-v1.5' -- comparable-or-better retrieval quality on
    benchmarks, no torch dependency. Any embeddings cached under the old
    model are NOT compatible with this one; call fit_corpus() again rather
    than reusing old cached vectors.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.corpus_embeddings = None

    @staticmethod
    def _normalize(matrix):
        """L2-normalize each row so dot product == cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return matrix / norms

    def fit_corpus(self, corpus_texts, batch_size=64):
        embeddings = np.array(list(self.model.embed(corpus_texts, batch_size=batch_size)))
        self.corpus_embeddings = self._normalize(embeddings)

    def get_scores(self, query_text):
        if self.corpus_embeddings is None:
            raise ValueError("Corpus embeddings not initialized. Call fit_corpus() first.")
        query_vector = np.array(list(self.model.embed([query_text])))
        query_vector = self._normalize(query_vector)[0]
        # corpus is already normalized, so dot product == cosine similarity
        return self.corpus_embeddings @ query_vector

    def get_ranking(self, query_text):
        scores = self.get_scores(query_text)
        return np.argsort(scores)[::-1].tolist(), scores