import os
import re
import sys
import psutil
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "src"))

from retrieval import BM25
# from retrieval import DenseRetriever, reciprocal_rank_fusion  # DISABLED: requires ~1.5 GB RAM
from cv_extractor import extract_text_from_file

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "data", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

PARQUET_PATH = os.path.join(BASE_DIR, "data", "processed", "jobs.parquet")

# --- RAM Tracker ---
_process = psutil.Process(os.getpid())


def get_ram_mb():
    """Return current process RAM usage in MB."""
    return _process.memory_info().rss / (1024 * 1024)


def log_ram(label=""):
    """Print current RAM usage with an optional label."""
    print(f"[RAM] {label}: {get_ram_mb():.1f} MB")


# --- Query preprocessing: MUST match preprocessing.py's process_nlp() exactly,
# otherwise BM25 term matching silently degrades (query tokens won't align
# with how the corpus was indexed -- e.g. "developers" in a query vs.
# "developer" in the lemmatized index).
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


print("Loading dataset into global memory...")
log_ram("before loading dataset")
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"Missing processed corpus at {PARQUET_PATH}. Run preprocessing.py first.")

df_corpus = pd.read_parquet(PARQUET_PATH)
corpus_tokens = [str(text).split() for text in df_corpus["text_processed"]]
log_ram("after loading dataset")

print("Building BM25 sparse index...")
log_ram("before BM25")
bm25_retriever = BM25(corpus_tokens)
log_ram("after BM25")

# DISABLED for Cloud Run free tier (512 MB RAM limit).
# Dense retriever loads fastembed ONNX Runtime which allocates ~1.5 GB
# regardless of model size. Re-enable when deploying to paid tier (2-4 GB).
# To enable hybrid search: uncomment below + switch search logic to use
# reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
#
# print("Loading Dense Retriever model (fastembed default: BAAI/bge-small-en-v1.5)...")
# log_ram("before dense retriever")
# dense_retriever = DenseRetriever()
# dense_corpus_texts = (df_corpus["title"].fillna("") + " " + df_corpus["text_translated_en"].fillna("")).tolist()
# dense_retriever.fit_corpus(dense_corpus_texts, batch_size=64)
# log_ram("after dense retriever")
print("Dense retriever DISABLED — using BM25 only. See limitation note in README.")
print("System ready. Starting web server...")


@app.route("/", methods=["GET", "POST"])
def index():
    query_text = ""
    display_query = ""
    results = []
    error_msg = None
    filter_no_german = request.form.get("filter_no_german") == "on"
    selected_exp = request.form.get("filter_experience", "All")

    if request.method == "POST":
        log_ram("before search")
        uploaded_file = request.files.get("cv_file")
        if uploaded_file and uploaded_file.filename != "":
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(file_path)
            try:
                query_text = extract_text_from_file(file_path)
                display_query = f"[Uploaded CV]: {filename}"
            except Exception as e:
                error_msg = f"Failed to extract text from file: {str(e)}"
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            query_text = request.form.get("query", "").strip()
            display_query = query_text

        if query_text and not error_msg:
            query_tokens = preprocess_query(query_text)
            bm25_ranking, bm25_scores = bm25_retriever.get_ranking(query_tokens)

            # DISABLED: Dense retrieval requires ~1.5 GB RAM.
            # To re-enable hybrid search, uncomment below and switch to
            # reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
            #
            # dense_query = query_text[:1000]
            # dense_ranking, _ = dense_retriever.get_ranking(dense_query)
            # hybrid_ranking, rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

            # BM25-only fallback for prototype
            ranking = bm25_ranking
            scores = bm25_scores

            for idx in ranking:
                row = df_corpus.iloc[idx]

                # Apply German Filter
                if filter_no_german and row["requires_german"]:
                    continue

                # Apply Seniority Filter
                job_exp = row.get("experience_level", "Mid")
                if selected_exp != "All" and job_exp != selected_exp:
                    continue

                results.append({
                    "title": row["title"],
                    "company": row["company"],
                    "location": row["location"],
                    "source": row["source"],
                    "url": row["url"],
                    "date_posted": str(row["date_posted"])[:10],
                    "requires_german": row["requires_german"],
                    "experience_level": job_exp,
                    "snippet": str(row["text_translated_en"])[:250] + "...",
                    "score": f"{scores[idx]:.4f}"
                })
                if len(results) >= 20:
                    break

        log_ram("after search")

    return render_template("index.html", query=display_query, results=results, error=error_msg, filter_no_german=filter_no_german, selected_exp=selected_exp)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)