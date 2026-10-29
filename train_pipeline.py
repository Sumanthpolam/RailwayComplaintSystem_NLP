"""
Railway Passenger Complaint Classification - Training Pipeline
----------------------------------------------------------------
Run this file whenever you get a new/updated dataset. It:

 1. Loads the raw messy CSV from data/Messy_DATA.csv
 2. Cleans the structured columns (names, coach codes, PNR, etc.)
 3. Applies NLP preprocessing to the complaint text (lowercase, noise
    removal, tokenization, stopword removal, lemmatization)
 4. Assigns a complaint TYPE label using rule-based keyword matching
    (the raw data has no ready-made label, so this step creates one)
 5. Splits the data 70% train / 30% test
 6. Vectorizes text with TF-IDF and trains a classifier
 7. Saves the trained model + vectorizer into models/ for the app to use

Usage:
    python train_pipeline.py
"""

import re
import string
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

RAW_PATH = "data/Messy_DATA.csv"
CLEANED_OUT_PATH = "data/After_NLP_Preprocessing.csv"
MODEL_PATH = "models/complaint_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# ==================================================================
# STEP 1 - LOAD
# ==================================================================
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

# ==================================================================
# STEP 2 - CLEAN STRUCTURED (NON-TEXT) COLUMNS
# ==================================================================
str_cols = df.select_dtypes(include="object").columns
for c in str_cols:
    df[c] = df[c].str.strip()

df["name"] = df["name"].str.title()
df["train_name"] = df["train_name"].str.title()
df["coach"] = df["coach"].str.upper()
df["pnr_number"] = df["pnr_number"].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
df["seat_number"] = df["seat_number"].astype("Int64")

for c in ["name", "pnr_number", "train_name", "coach"]:
    df[c] = df[c].fillna("Unknown")

before = len(df)
df = df.drop_duplicates(subset=["pnr_number", "complaint"], keep="first")
print(f"Removed {before - len(df)} duplicate rows")

# ==================================================================
# STEP 3 - NLP TEXT PREPROCESSING
# ==================================================================
NEGATIONS = {"no", "not", "nor", "never", "none", "cannot", "without"}
STOPWORDS = set("a an the and or but if then else when while of at by for with about against between into through during before after above below to from up down in out on off over under again further once here there all any both each few more most other some such only own same so than too very can will just should now is am are was were be been being have has had do does did this that these those it its i me my we our you your he she they them their what which who whom".split()) - NEGATIONS


def normalize_token(token):
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_lemmatize(text):
    tokens = re.findall(r"[a-z]+", text)
    return [normalize_token(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


df["complaint_clean"] = df["complaint"].apply(clean_text)
df["complaint_tokens"] = df["complaint_clean"].apply(tokenize_lemmatize)
df["complaint_processed"] = df["complaint_tokens"].apply(lambda toks: " ".join(toks))
df = df[df["complaint_processed"].str.len() > 0]

# ==================================================================
# STEP 4 - RULE-BASED COMPLAINT TYPE LABELING
# ------------------------------------------------------------------
# This dataset has no ground-truth complaint_type column, so we
# generate labels using keyword rules on the ORIGINAL complaint text
# (checked in a fixed priority order so more specific / urgent
# categories are matched before generic ones).
# ==================================================================
KEYWORD_RULES = {
    "Security": [
        "unsafe", "danger", "threat", "suspicious", "theft", "stolen", "steal",
        "harass", "safety", r"\bfire\b", "smoke", "molest", "assault",
        "security", "causing trouble", "immediate security",
    ],
    "Toilet/Sanitation": [
        "toilet", "washroom", "restroom", "sanitation", "flush", "bathroom",
        "sleeping area",
    ],
    "Food": [
        "food", "meal", "catering", "rice", "stale", "spoiled", "rotten",
        "tasteless", "unhygienic", "contaminated", "food quality", "food poisoning",
        "cold food", "poor quality food", "packed", "snack", "breakfast", "lunch", "dinner",
        "served cold", "small portion", "portion was", "fresh",
    ],
    "AC/Electrical": [
        r"\bac\b", "air condition", "cooling", "electrical", "electricity",
        "power failure", r"\bfan\b", r"\blight\b", "plug", "socket",
    ],
    "Water": [
        "water dispenser", "water supply", "no water", "water leakage",
        "drinking water", "shortage of water", "water tank",
        "supplying water", "provide water", r"\bwater\b",
    ],
    "Cleanliness": [
        "dirty", "dust", "garbage", "unclean", "waste", r"\bsmell\b", "stain",
        "cockroach", "trash", "not been cleaned", "cleaned properly",
        "covered with dirt", "not clean", "needs cleaning",
    ],
    "Staff/Service": [
        "staff", "rude", "disrespect", "refused", "refuse", "misbehav",
        "behaviour", "behavior", "attendant", "employee ignored",
        "did not respond",
    ],
    "Delay": [
        "delay", r"\blate\b", "waiting", "schedule", "departed on time",
        "running late", "standing at the station", "reached the station",
    ],
    "Reservation/Seat": [
        r"\bseat\b", "berth", "reserved", "reservation", r"\bpnr\b",
        "incorrect seat", "already occupied", "seat number is not available",
    ],
    "Luggage/Lost Property": [
        "luggage", r"\bbag\b", "suitcase", "missing", "lost property",
        "belongings", "lost my bag", "find my bag", "find my luggage",
    ],
}
CATEGORY_ORDER = list(KEYWORD_RULES.keys())


def normalize_for_rules(text):
    t = str(text).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def assign_complaint_type(raw_text):
    t = normalize_for_rules(raw_text)
    for category in CATEGORY_ORDER:
        for pattern in KEYWORD_RULES[category]:
            if re.search(pattern, t):
                return category
    return "Other"


df["complaint_type"] = df["complaint"].apply(assign_complaint_type)
print("\nComplaint type distribution:\n", df["complaint_type"].value_counts())

# ==================================================================
# STEP 5 - SAVE THE CLEANED + LABELED DATASET
# ==================================================================
final_cols = [
    "passenger_id", "name", "pnr_number", "train_number", "train_name",
    "coach", "seat_number", "complaint", "complaint_clean",
    "complaint_processed", "complaint_tokens", "complaint_type",
]
df_out = df[final_cols].reset_index(drop=True)
df_out.to_csv(CLEANED_OUT_PATH, index=False)
print(f"\nCleaned + labeled dataset saved -> {CLEANED_OUT_PATH}")

# ==================================================================
# STEP 6 - TRAIN / TEST SPLIT (70/30)
# ==================================================================
X_text = df["complaint_processed"]
y = df["complaint_type"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train_text)}  Test size: {len(X_test_text)}")

# ==================================================================
# STEP 7 - TF-IDF + MODEL TRAINING (compares 3 models, keeps the best)
# ==================================================================
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

candidates = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

best_name, best_model, best_acc = None, None, -1
for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print(f"\nBest model: {best_name} (accuracy = {best_acc:.4f})")

# ==================================================================
# STEP 8 - SAVE MODEL + VECTORIZER FOR THE STREAMLIT APP
# ==================================================================
joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"\nSaved model      -> {MODEL_PATH}")
print(f"Saved vectorizer -> {VECTORIZER_PATH}")
print("\nTraining complete. You can now run: streamlit run app.py")
