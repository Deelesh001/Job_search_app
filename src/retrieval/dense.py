import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever:
    """Multilingual sentence embeddings for dense vector retrieval."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None

    def fit_corpus(self, corpus_texts, batch_size=64):
        self.corpus_embeddings = self.model.encode(
            corpus_texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )

    def get_scores(self, query_text):
        if self.corpus_embeddings is None:
            raise ValueError("Corpus embeddings not initialized. Call fit_corpus() first.")
        query_vector = self.model.encode([query_text], normalize_embeddings=True)
        return cosine_similarity(query_vector, self.corpus_embeddings)[0]

    def get_ranking(self, query_text):
        scores = self.get_scores(query_text)
        return np.argsort(scores)[::-1].tolist(), scores