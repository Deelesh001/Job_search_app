---
title: Job Information Retrieval System
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Hybrid Job Information Retrieval System

An enterprise-grade semantic search and CV matching engine combining **BM25 sparse retrieval** with **Dense vector embeddings (`all-MiniLM-L6-v2`)**, synthesized via **Reciprocal Rank Fusion (RRF)**. Designed specifically for cross-lingual (English/German) job market analysis and resume parsing.

---

## Key Features

* **Hybrid Search Architecture:** Integrates lexical term-matching (BM25) with dense neural semantic search, merged seamlessly using Reciprocal Rank Fusion (RRF).
* **Multi-Format Resume Parser:** Supports direct uploading of `.pdf`, `.docx`, and `.txt` documents via an automated text extraction extraction pipeline (`pdfplumber` and `python-docx`).
* **Cross-Lingual Capability:** Translates German job listings into English at ingestion time using Argos Translate to maintain a unified semantic vector space.
* **Parametric Metadata Filtering:** Includes hard-constraint filters for **Seniority Levels** (Entry, Mid, Senior) and **Language Proficiency** (German filtering) to prevent mismatched search results.
* **Production-Ready WSGI Server:** Configured with Gunicorn using preloaded memory sharing (`--preload`) and multi-threading to handle concurrent web traffic efficiently.

---

## Project Structure

```text
job-ir-system/
├── app.py                     # Main Flask web application
├── Dockerfile                 # Container blueprint for Hugging Face Spaces
├── requirements.txt           # Python dependency manifests
├── README.md                  # Project documentation & metadata
├── data/
│   ├── processed/
│   │   └── jobs.parquet       # Preprocessed, tokenized, and indexed corpus
│   └── uploads/               # Temporary directory for uploaded CV files
├── eval/
│   └── evaluate.py            # Information retrieval evaluation metrics script
├── src/
│   ├── cv_extractor.py        # Unified parser for PDF, DOCX, and TXT files
│   ├── ingestion.py           # Scraping and raw data ingestion pipeline
│   ├── preprocessing.py       # NLP tokenization, lemmatization, and translation
│   └── retrieval/
│       ├── bm25.py            # Lexical sparse retriever implementation
│       ├── dense.py           # Sentence-Transformers dense vector retriever
│       └── hybrid.py          # Reciprocal Rank Fusion (RRF) algorithm
└── templates/
    └── index.html             # Responsive web UI with dynamic filters