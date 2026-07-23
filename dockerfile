FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/home/user

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN mkdir -p data/uploads data/processed && \
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--preload", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]
