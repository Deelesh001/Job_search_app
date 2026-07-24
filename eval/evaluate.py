import os
import sys
import json
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import os
# Must be set BEFORE importing PyTorch, PySpark, or Argos Translate to prevent macOS C++ deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Connect module paths to match project directory structure
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from retrieval.bm25 import BM25
from retrieval.dense import DenseRetriever
from retrieval.hybrid import reciprocal_rank_fusion

PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "jobs.parquet")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark")
BENCHMARK_PATH = os.path.join(BENCHMARK_DIR, "queries.json")

# --- Query preprocessing: MUST match preprocessing.py's process_nlp() and
# app.py's preprocess_query() exactly. If evaluation tokenizes queries
# differently than the live app does, the P@10/R@10 numbers you report
# don't actually reflect what a user experiences during the demo.
for pkg in ['wordnet', 'omw-1.4', 'stopwords']:
    try:
        nltk.data.find(f'corpora/{pkg}.zip')
    except LookupError:
        nltk.download(pkg, quiet=True)

_stop_words_set = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def preprocess_query(text):
    if not text or not isinstance(text, str):
        return []
    tokens = re.findall(r"\w+", text.lower())
    return [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words_set and len(t) > 1]


def ensure_benchmark_data():
    """Creates the 5 benchmark queries + relevance definitions if missing."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    if os.path.exists(BENCHMARK_PATH):
        return

    queries = [
        {"query": "Data Scientist Python Machine Learning", "must_have": ["python"], "nice_to_have": ["data", "scientist", "machine", "learning", "ai", "ml"]},
        {"query": "Remote Backend Java Developer Spring", "must_have": ["java"], "nice_to_have": ["spring", "backend", "developer", "engineer", "remote"]},
        {"query": "DevOps Engineer AWS Kubernetes Docker", "must_have": [], "nice_to_have": ["devops", "kubernetes", "k8s", "aws", "docker", "cloud", "terraform"]},
        {"query": "Frontend React TypeScript Web Developer", "must_have": [], "nice_to_have": ["react", "typescript", "javascript", "frontend", "vue", "angular", "web"]},
        {"query": "German speaking Project Manager", "must_have": [], "nice_to_have": ["project", "manager", "management", "agile", "scrum", "product", "german", "deutsch"]}
    ]
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
    print(f"Generated benchmark queries at {BENCHMARK_PATH}")


def get_ground_truth_relevance(df, query_def):
    req_all = query_def.get("must_have", [])
    req_any = query_def.get("nice_to_have", [])

    relevant_indices = set()
    for idx, row in df.iterrows():
        text = f"{row['title']} {row['text_translated_en']}".lower()
        if req_all and not all(re.search(r"\b" + re.escape(t) + r"\b", text) for t in req_all):
            continue
        if req_any and not any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in req_any):
            continue
        relevant_indices.add(idx)
    return relevant_indices


def calculate_metrics(retrieved_indices, relevant_indices, k=10):
    if not relevant_indices:
        return 0.0, 0.0
    top_k = set(retrieved_indices[:k])
    true_positives = len(top_k.intersection(relevant_indices))
    precision = true_positives / float(k)
    recall = true_positives / float(len(relevant_indices))
    return precision, recall


def main():
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing preprocessed corpus at {PARQUET_PATH}. Run preprocessing.py first.")

    ensure_benchmark_data()

    print("Loading preprocessed corpus...")
    df = pd.read_parquet(PARQUET_PATH)

    print("Initializing modular BM25 sparse index...")
    corpus_tokens = [str(text).split() for text in df["text_processed"]]
    bm25 = BM25(corpus_tokens)

    print("Initializing modular DenseRetriever (fastembed default: BAAI/bge-small-en-v1.5) and encoding corpus...")
    dense_retriever = DenseRetriever()
    dense_corpus_texts = (df["title"].fillna("") + " " + df["text_translated_en"].fillna("")).tolist()
    dense_retriever.fit_corpus(dense_corpus_texts, batch_size=64)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print("\nRunning Evaluation Benchmark across Modular Architecture...\n")
    results = []

    for tq in test_queries:
        query_text = tq["query"]
        relevant_idx = get_ground_truth_relevance(df, tq)

        # 1. Sparse -- tokenized the same way the corpus was (lemmatized, stopwords removed)
        query_tokens = preprocess_query(query_text)
        bm25_ranking, _ = bm25.get_ranking(query_tokens)
        p_bm25, r_bm25 = calculate_metrics(bm25_ranking, relevant_idx, k=10)

        # 2. Dense
        dense_ranking, _ = dense_retriever.get_ranking(query_text)
        p_dense, r_dense = calculate_metrics(dense_ranking, relevant_idx, k=10)

        # 3. Hybrid RRF
        hybrid_ranking, _ = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
        p_hybrid, r_hybrid = calculate_metrics(hybrid_ranking, relevant_idx, k=10)

        results.append({
            "Query": query_text[:30],
            "Relevant Docs": len(relevant_idx),
            "BM25 P@10": f"{p_bm25:.2f}",
            "BM25 R@10": f"{r_bm25:.2f}",
            "Dense P@10": f"{p_dense:.2f}",
            "Dense R@10": f"{r_dense:.2f}",
            "Hybrid P@10": f"{p_hybrid:.2f}",
            "Hybrid R@10": f"{r_hybrid:.2f}"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("\nModular evaluation complete. Architecture validated.")


if __name__ == "__main__":
    main()