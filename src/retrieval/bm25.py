import numpy as np

class BM25:
    """Hand-coded BM25 indexing and scoring without external dependencies."""
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lengths = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = self.doc_lengths.mean() if self.corpus_size > 0 else 0.0
        
        self.doc_freqs = {}
        self.term_freqs = []
        for doc in corpus_tokens:
            tf = {}
            for word in doc:
                tf[word] = tf.get(word, 0) + 1
            for word in tf.keys():
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
            self.term_freqs.append(tf)
            
        self.idf = {}
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            
    def get_scores(self, query_tokens):
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for i, tf_map in enumerate(self.term_freqs):
                tf = tf_map.get(word, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[i] / (self.avgdl + 1e-9)))
                scores[i] += idf_val * (tf * (self.k1 + 1.0)) / (denom + 1e-9)
        return scores

    def get_ranking(self, query_tokens):
        scores = self.get_scores(query_tokens)
        return np.argsort(scores)[::-1].tolist(), scores