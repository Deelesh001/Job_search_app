import os
import re
import ssl
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup
import langdetect
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import argostranslate.package
import argostranslate.translate

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    ctx = ssl._create_unverified_context()
except AttributeError:
    pass
else:
    ssl._create_default_https_context = lambda: ctx
    https_handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.build_opener(https_handler)
    urllib.request.install_opener(opener)

for pkg in ['wordnet', 'omw-1.4', 'stopwords']:
    try:
        nltk.data.find(f'corpora/{pkg}.zip')
    except LookupError:
        nltk.download(pkg, quiet=True)

BASE_DIR = os.path.dirname(__file__)
RAW_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "raw_jobs_snapshot.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "jobs.parquet")
TRANSLATION_CACHE_PATH = os.path.join(OUTPUT_DIR, "translated_cache.parquet")


def setup_translation():
    installed = argostranslate.translate.get_installed_languages()
    de_lang = next((x for x in installed if x.code == "de"), None)
    en_lang = next((x for x in installed if x.code == "en"), None)
    if de_lang and en_lang:
        return de_lang.get_translation(en_lang)
    try:
        argostranslate.package.update_package_index()
        packages = argostranslate.package.get_available_packages()
        pkg = next((x for x in packages if x.from_code == "de" and x.to_code == "en"), None)
        if pkg:
            argostranslate.package.install_from_path(pkg.download())
            installed = argostranslate.translate.get_installed_languages()
            de_lang = next((x for x in installed if x.code == "de"), None)
            en_lang = next((x for x in installed if x.code == "en"), None)
    except Exception as e:
        print(f"Translation model setup failed: {e}")
        return None
    return de_lang.get_translation(en_lang) if (de_lang and en_lang) else None


def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def get_lang(text):
    if not text or len(str(text)) < 10:
        return "en"
    try:
        return langdetect.detect(str(text))
    except Exception:
        return "en"


def check_german_req(text):
    if not text:
        return False
    keywords = [
        r"deutschkenntnisse", r"german b1", r"german b2", r"german c1", r"german c2",
        r"fließend deutsch", r"verhandlungssicher", r"deutsch zwingend",
        r"good german", r"fluent german", r"german language skills"
    ]
    return bool(re.search("|".join(keywords), str(text).lower()))


def classify_experience(text):
    """Classifies jobs into Entry, Mid, or Senior based on titles and required years."""
    if not text:
        return "Mid"
    text_lower = str(text).lower()
    
    # 1. Check Senior/Lead keywords and high year requirements (5+ years)
    senior_patterns = [
        r"\b(senior|lead|principal|head of|director|vp|chief|staff engineer)\b",
        r"\b([5-9]|1[0-9])\s*\+?\s*(?:years?|jahren?)\b",
        r"\b(?:at least|minimum)\s*([5-9]|1[0-9])\s*(?:years?|jahren?)\b"
    ]
    if any(re.search(pat, text_lower) for pat in senior_patterns):
        return "Senior"
        
    # 2. Check Entry-level / Student / Junior keywords and 0-2 years
    entry_patterns = [
        r"\b(junior|entry[\s-]level|intern|internship|trainee|working student|werkstudent|graduate)\b",
        r"\b(?:0|1|2)\s*(?:-|to)?\s*(?:1|2)?\s*(?:years?|jahren?)\b",
        r"\bno experience\b"
    ]
    if any(re.search(pat, text_lower) for pat in entry_patterns):
        return "Entry"
        
    # 3. Default remaining roles to Mid-level
    return "Mid"


def translate_corpus(df, translator):
    n_de = (df["lang"] == "de").sum()
    print(f"Language breakdown:\n{df['lang'].value_counts()}")
    print(f"Translating {n_de} German-tagged postings (snippet-only, 350 chars)...")
    translated = []
    count = 0
    for _, row in df.iterrows():
        if row["lang"] == "de" and translator and row["text_original"]:
            try:
                translated.append(translator.translate(row["text_original"][:350]))
                count += 1
                if count % 25 == 0:
                    print(f"  Translated {count}/{n_de}")
            except Exception:
                translated.append(row["text_original"])
        else:
            translated.append(row["text_original"])
    df["text_translated_en"] = translated
    return df


def dedup_cross_source(df):
    before = len(df)
    clean_title = df["title"].fillna("").astype(str).str.lower().str.strip()
    clean_company = df["company"].fillna("").astype(str).str.lower().str.strip()
    df["_dedup_key"] = clean_title + "||" + clean_company
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"]).reset_index(drop=True)
    print(f"Cross-source dedup: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing raw data at {RAW_PATH}. Run ingestion.py first.")

    cache_is_valid = (
        os.path.exists(TRANSLATION_CACHE_PATH) and 
        os.path.getmtime(TRANSLATION_CACHE_PATH) >= os.path.getmtime(RAW_PATH)
    )

    if cache_is_valid:
        print(f"Found valid translation cache at {TRANSLATION_CACHE_PATH}, loading...")
        df = pd.read_parquet(TRANSLATION_CACHE_PATH)
    else:
        print("Step 1: Loading raw data and running driver-side NLP cleaning...")
        df = pd.read_json(RAW_PATH)

        expected_cols = ["title", "company", "location", "description", "source", "url", "date_posted", "category"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        df["clean_desc"] = df["description"].apply(clean_html)
        df["text_original"] = df["title"].apply(clean_html) + " " + df["clean_desc"]

        df = dedup_cross_source(df)

        df["lang"] = df["text_original"].apply(get_lang)
        df["requires_german"] = df["text_original"].apply(check_german_req)

        translator = setup_translation()
        df = translate_corpus(df, translator)

        df.to_parquet(TRANSLATION_CACHE_PATH, index=False)
        print(f"Cached translated corpus to {TRANSLATION_CACHE_PATH}")

    print("Step 2: Running native NLP tokenization, experience tagging, and lemmatization...")
    str_cols = ["title", "company", "location", "text_original", "text_translated_en",
                "source", "url", "date_posted", "category", "lang"]
    
    for col in str_cols:
        if col not in df.columns:
            df[col] = ""
            
    df[str_cols] = df[str_cols].fillna("")
    df["requires_german"] = df["requires_german"].fillna(False).astype(bool)
    
    # Apply experience level classification
    df["experience_level"] = df["text_original"].apply(classify_experience)
    print(f"Experience level breakdown:\n{df['experience_level'].value_counts()}")

    stop_words_set = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def process_nlp(text):
        if not text or not isinstance(text, str):
            return ""
        tokens = re.findall(r"\w+", text.lower())
        cleaned = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words_set and len(t) > 1]
        return " ".join(cleaned)

    df["text_processed"] = df["text_translated_en"].apply(process_nlp)

    final_cols = str_cols + ["requires_german", "experience_level", "text_processed"]
    df_final = df[final_cols]

    print("Step 3: Exporting to Parquet...")
    df_final.to_parquet(OUTPUT_PATH, index=False)
    print(f"Success! Preprocessed data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()