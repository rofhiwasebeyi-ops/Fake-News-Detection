import streamlit as st
from lime.lime_text import LimeTextExplainer
from pathlib import Path
import io
import PyPDF2
import docx
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import hashlib
import warnings
warnings.filterwarnings("ignore")


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load Models 
@st.cache_resource
def load_models():
    meta_model = joblib.load("models/meta_classifier.pkl")
    vectorizer = joblib.load("models/tf-idf_vectorizer.pkl")

    ml_models = {}
    base_model_names = [
        "Logistic_Regression",
        "Linear_SVC",
        "Multinomial_NB",
        "Random_Forest",
        "Gradient_Boosting"
    ]
    for name in base_model_names:
        ml_models[name] = joblib.load(f"models/{name}_tuned_model.pkl")

    bert_tokenizer = DistilBertTokenizerFast.from_pretrained(
        "models/fine_tuned_distilbert_fake_news")
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        "models/fine_tuned_distilbert_fake_news").to(device)
    bert_model.eval()

    return meta_model, vectorizer, ml_models, bert_tokenizer, bert_model


meta_model, vectorizer, ml_models, bert_tokenizer, bert_model = load_models()
MAX_BERT_LEN = 512


# Helper Functions 
def get_ml_probs(text: str):
    X_vec = vectorizer.transform([text])
    probs = []
    for name, model in ml_models.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_vec)[:, 1][0]
        else:
            score = model.decision_function(X_vec)[0]
            p = 1 / (1 + np.exp(-score))
        probs.append(0.5 if np.isnan(p) else p)
    return probs


def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


def get_bert_prob_chunked(text: str) -> float:
    chunks = chunk_text(text, max_words=200)
    probs_list = []
    for chunk in chunks:
        if not chunk.strip():
            probs_list.append(0.5)
            continue
        inputs = bert_tokenizer(chunk, return_tensors="pt",
                                truncation=True, padding=True, max_length=MAX_BERT_LEN).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            probs_list.append(probs[1] if not np.isnan(probs[1]) else 0.5)
    return float(np.mean(probs_list))


def ensemble_predict_chunked(text: str):
    ml_probs = get_ml_probs(text)
    bert_prob = get_bert_prob_chunked(text)

    meta_features = np.array([ml_probs + [bert_prob]])
    probs = meta_model.predict_proba(meta_features)[0]
    pred = np.argmax(probs)
    confidence = float(probs[pred])

    # Flip probabilities if prediction is FAKE (class 0)
    if pred == 0:
        ml_probs = [1 - p for p in ml_probs]
        bert_prob = 1 - bert_prob

    model_names = list(ml_models.keys())
    details = {name: prob for name, prob in zip(model_names, ml_probs)}
    details["BERT"] = bert_prob
    details["Meta"] = probs.tolist()

    return pred, confidence, details


def extract_text_from_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    if ext == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif ext == ".docx":
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""


def lime_explain_meta_chunked(text, top_n_features=7, num_samples=20):
    """
    Generate LIME explanations for the meta model on chunked text.
    Reduced num_samples for speed.
    """
    chunks = chunk_text(text, max_words=200)

    def pred_fn(x):
        probs_list = []
        for t in x:
            t = t.strip()
            if not t:
                probs_list.append(np.array([0.5, 0.5]))
                continue
            ml_probs = get_ml_probs(t)
            bert_prob = get_bert_prob_chunked(t)
            ml_probs = [0.5 if np.isnan(p) else p for p in ml_probs]
            bert_prob = 0.5 if np.isnan(bert_prob) else bert_prob
            meta_input = np.array([ml_probs + [bert_prob]])
            probs_list.append(meta_model.predict_proba(meta_input)[0])
        return np.array(probs_list)

    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
    feature_weights = {}

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        exp = explainer.explain_instance(chunk, pred_fn, num_features=top_n_features, num_samples=num_samples)
        for word, weight in exp.as_list():
            feature_weights[word] = feature_weights.get(word, 0) + weight

    sorted_features = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n_features]

    class ExpWrapper:
        def as_list(self_inner):
            return top_features

    return ExpWrapper()


def display_lime_as_table(exp):
    df_lime = pd.DataFrame(exp.as_list(), columns=["Word", "Weight"])
    st.table(df_lime)
    words, weights = zip(*exp.as_list())
    colors = ["green" if w > 0 else "red" for w in weights]
    plt.figure(figsize=(8, 4))
    plt.barh(words, weights, color=colors)
    plt.xlabel("Contribution to Prediction")
    plt.title("Top Words Affecting Meta Prediction")
    st.pyplot(plt)


def get_text_hash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_file_hash(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


# Streamlit UI
st.title("📰 Fake News Detector")

option = st.radio("Input method:", ["Enter Text", "Upload Document"])
text_input = ""
uploaded_file = None

if option == "Enter Text":
    text_input = st.text_area("Enter a news article:", height=200)
elif option == "Upload Document":
    uploaded_file = st.file_uploader("Upload a document (txt, pdf, docx)", type=["txt", "pdf", "docx"])
    if uploaded_file:
        text_input = extract_text_from_file(uploaded_file)
        st.success("✅ Document uploaded and text extracted!")
        st.text_area("Preview of extracted text:", text_input[:2000], height=200)

if st.button("Predict"):
    # Determine unique hash for caching
    if uploaded_file:
        unique_hash = get_file_hash(uploaded_file)
        text_data = extract_text_from_file(uploaded_file)
    elif text_input.strip():
        text_data = text_input.strip()
        unique_hash = get_text_hash(text_data)
    else:
        st.warning("Please enter text or upload a file.")
        st.stop()

    if "predictions_cache" not in st.session_state:
        st.session_state.predictions_cache = {}

    if unique_hash in st.session_state.predictions_cache:
        st.info("🔁 Using cached prediction — identical input detected!")
        overall_pred, overall_conf, details, exp_meta = st.session_state.predictions_cache[unique_hash]
    else:
        overall_pred, overall_conf, details = ensemble_predict_chunked(text_data)
        exp_meta = lime_explain_meta_chunked(text_data, top_n_features=10)
        st.session_state.predictions_cache[unique_hash] = (overall_pred, overall_conf, details, exp_meta)

    # Display
    color_banner = "green" if overall_pred == 1 else "red"
    st.markdown(
        f"<h2 style='color:white;background-color:{color_banner};padding:10px'>"
        f"Overall Prediction: {'REAL' if overall_pred == 1 else 'FAKE'} "
        f"(confidence: {overall_conf:.2f})</h2>",
        unsafe_allow_html=True
    )

    st.subheader("📊 Model Probabilities")
    for name, prob in details.items():
        if name != "Meta":
            st.write(f"**{name}:** {prob:.2f}")
    st.write(f"**Meta (Stacked Ensemble):** {details['Meta']}")

    st.subheader("🧠 LIME Explanation (Meta Model)")
    display_lime_as_table(exp_meta)

    # CSV download
    df = pd.DataFrame([{
        "Text": text_data[:200] + "..." if len(text_data) > 200 else text_data,
        "Prediction": "REAL" if overall_pred == 1 else "FAKE",
        "Confidence": overall_conf,
        **details
    }])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Prediction CSV", data=csv_buffer.getvalue(),
                       file_name="prediction_results.csv", mime="text/csv")

st.markdown("---")
st.caption("Powered by scikit-learn + DistilBERT + Streamlit + LIME")
