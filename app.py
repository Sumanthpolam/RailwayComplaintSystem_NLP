"""
Railway Complaint Classification System - Streamlit App
-----------------------------------------------------------
Run with:  streamlit run app.py

User types a complaint, presses Predict, and instantly sees:
  - Complaint Type   (predicted by the trained ML model)
  - Priority         (rule-based: Critical / High / Medium / Low)
  - Department       (mapped from complaint type)
  - Assigned To      (staff role mapped from complaint type)
"""

import re
import string
import joblib
import streamlit as st

# ==================================================================
# LOAD TRAINED MODEL + VECTORIZER (created by train_pipeline.py)
# ==================================================================
model = joblib.load("models/complaint_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

NEGATIONS = {"no", "not", "nor", "never", "none", "cannot", "without"}
STOPWORDS = set("a an the and or but if then else when while of at by for with about against between into through during before after above below to from up down in out on off over under again further once here there all any both each few more most other some such only own same so than too very can will just should now is am are was were be been being have has had do does did this that these those it its i me my we our you your he she they them their what which who whom".split()) - NEGATIONS


def normalize_token(token):
    # Lightweight suffix normalization; keeps preprocessing usable offline.
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


# ==================================================================
# TEXT PREPROCESSING (must match training-time preprocessing exactly)
# ==================================================================
def preprocess_text(text):
    text = str(text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[a-z]+", text)
    tokens = [normalize_token(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def rule_based_complaint_type(raw_text):
    """Route clear, high-signal complaints before using the ML fallback."""
    text = str(raw_text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Food quality/safety must be checked before generic cleanliness words.
    food_patterns = [
        r"\b(rice|food|meal|catering|stale|spoiled|rotten|tasteless|unhygienic|contaminated)\b",
        r"food poisoning", r"poor quality food", r"served cold", r"food is cold",
        r"food contains", r"insects? in (the )?(food|meal|rice)", r"(food|meal|rice).{0,25}(bad|smell|dirty|quality|fresh|cold)",
        r"(bad|poor|stale|spoiled|rotten).{0,20}(food|meal|rice)",
    ]
    if any(re.search(pattern, text) for pattern in food_patterns):
        return "Food"

    rules = {
        "Security": [r"\b(unsafe|danger|threat|theft|stolen|harass|security|fire|smoke|assault|molest)\b"],
        "Toilet/Sanitation": [r"\b(toilet|washroom|restroom|bathroom|flush|sanitation)\b"],
        "AC/Electrical": [r"\b(ac|fan|light|electricity|electrical|socket|plug)\b", r"air condition", r"power failure", r"not cooling"],
        "Water": [r"\bwater\b", r"water supply", r"water leakage"],
        "Cleanliness": [r"\b(dirty|dust|garbage|unclean|waste|stain|cockroach|trash|cleaning)\b", r"not clean", r"not been cleaned"],
        "Staff/Service": [r"\b(staff|rude|disrespect|misbehav|attendant|unprofessional)\b", r"refused to", r"did not respond"],
        "Delay": [r"\b(delay|delayed|late|waiting)\b", r"running late"],
        "Reservation/Seat": [r"\b(seat|berth|reserved|reservation|pnr)\b"],
        "Luggage/Lost Property": [r"\b(luggage|bag|suitcase|belongings)\b", r"lost property"],
    }
    for category, patterns in rules.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return category
    return None


def predict_complaint_type(text):
    # Strong keyword matches prevent obvious complaints (e.g. rice/food quality)
    # from being misrouted by the statistical model.
    rule_prediction = rule_based_complaint_type(text)
    if rule_prediction is not None:
        return rule_prediction
    cleaned = preprocess_text(text)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]


# ==================================================================
# PRIORITY DETECTION (rule-based)
# ==================================================================
def predict_priority(text, complaint_type):
    text = text.lower()

    critical_words = [
        "fire", "smoke", "theft", "stolen", "robbery", "harassment",
        "attack", "fight", "injury", "accident", "danger", "unsafe",
        "emergency",
    ]
    high_words = [
        "delay", "delayed", "no water", "water leakage", "power failure",
        "security",
    ]

    if any(word in text for word in critical_words):
        return "Critical"
    elif any(word in text for word in high_words):
        return "High"
    elif complaint_type in ["AC/Electrical", "Food", "Water", "Staff/Service", "Security"]:
        return "Medium"
    else:
        return "Low"


# ==================================================================
# DEPARTMENT + PERSON ROUTING
# ==================================================================
DEPARTMENT_MAP = {
    "AC/Electrical": "Electrical Department",
    "Food": "Catering Department",
    "Cleanliness": "Housekeeping Department",
    "Toilet/Sanitation": "Housekeeping Department",
    "Delay": "Operations Department",
    "Water": "Water & Sanitation Department",
    "Security": "Railway Security",
    "Staff/Service": "Passenger Services",
    "Reservation/Seat": "Reservation Department",
    "Luggage/Lost Property": "Lost & Found Office",
    "Other": "General Complaint Cell",
}

PERSON_MAP = {
    "AC/Electrical": "Electrician",
    "Food": "Catering Supervisor",
    "Cleanliness": "Housekeeping Staff",
    "Toilet/Sanitation": "Housekeeping Staff",
    "Delay": "Operations Controller",
    "Water": "Water Maintenance Staff",
    "Security": "Railway Security Officer",
    "Staff/Service": "Passenger Service Officer",
    "Reservation/Seat": "TTE / Reservation Officer",
    "Luggage/Lost Property": "Lost & Found Officer",
    "Other": "Complaint Cell Officer",
}


def assign_department(complaint_type):
    return DEPARTMENT_MAP.get(complaint_type, "General Complaint Cell")


def assign_person(complaint_type):
    return PERSON_MAP.get(complaint_type, "Complaint Cell Officer")


# ==================================================================
# STREAMLIT UI
# ==================================================================
st.set_page_config(
    page_title="Railway Complaint System",
    page_icon="🚆",
    layout="centered",
)

st.title("🚆 Railway Complaint Classification System")
st.write("Enter a railway passenger complaint below to automatically classify, "
         "prioritize, and route it to the right department.")

complaint = st.text_area(
    "Enter Complaint",
    placeholder="Example: AC is not working in coach S1",
    height=120,
)

if st.button("Predict", type="primary"):
    if complaint.strip() == "":
        st.warning("Please enter a complaint before predicting.")
    else:
        complaint_type = predict_complaint_type(complaint)
        priority = predict_priority(complaint, complaint_type)
        department = assign_department(complaint_type)
        person = assign_person(complaint_type)

        st.subheader("Prediction Result")

        priority_color = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢",
        }.get(priority, "⚪")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Complaint Type", complaint_type)
            st.metric("Priority", f"{priority_color} {priority}")
        with col2:
            st.metric("Department", department)
            st.metric("Assigned To", person)

st.markdown("---")
st.caption("Railway Passenger Complaint Classification & Routing System")
