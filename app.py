import os
import sys
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "src"))

from retrieval import BM25, DenseRetriever, reciprocal_rank_fusion
from cv_extractor import extract_text_from_file

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "data", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

PARQUET_PATH = os.path.join(BASE_DIR, "data", "processed", "jobs.parquet")

print("Loading dataset into global memory...")
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"Missing processed corpus at {PARQUET_PATH}. Run preprocessing.py first.")

df_corpus = pd.read_parquet(PARQUET_PATH)
corpus_tokens = [str(text).split() for text in df_corpus["text_processed"]]

print("Building BM25 sparse index...")
bm25_retriever = BM25(corpus_tokens)

print("Loading Dense Retriever model ('all-MiniLM-L6-v2')...")
dense_retriever = DenseRetriever("all-MiniLM-L6-v2")
dense_corpus_texts = (df_corpus["title"].fillna("") + " " + df_corpus["text_translated_en"].fillna("")).tolist()
dense_retriever.fit_corpus(dense_corpus_texts, batch_size=64)
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
            query_tokens = query_text.lower().split()
            bm25_ranking, _ = bm25_retriever.get_ranking(query_tokens)

            dense_query = query_text[:1000]
            dense_ranking, _ = dense_retriever.get_ranking(dense_query)

            hybrid_ranking, rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

            for idx in hybrid_ranking:
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
                    "score": f"{rrf_scores[idx]:.4f}"
                })
                if len(results) >= 20:
                    break

    return render_template("index.html", query=display_query, results=results, error=error_msg, filter_no_german=filter_no_german, selected_exp=selected_exp)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)