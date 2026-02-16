import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import warnings
warnings.filterwarnings("ignore")


# Load Dataset
df = pd.read_csv("data/preprocessed/preprocessed_dataset.csv", na_filter=False)
df = df[df["clean_content"].notna() & (df["clean_content"].str.strip() != '')]

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])


# Helper: Get Probabilities from Sklearn Models
def get_model_probs(texts, model_name, vec_path="models/tf-idf_vectorizer.pkl"):
    """Get prediction probabilities or scores for a given model."""
    model_path = f"models/{model_name}_tuned_model.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    X_vec = vectorizer.transform(texts)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_vec)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_vec)
        return 1 / (1 + np.exp(-scores))  # sigmoid normalization
    else:
        preds = model.predict(X_vec)
        return preds.astype(float)


# Helper: Get Probabilities from DistilBERT
def get_bert_probs(texts, model_path="models/fine_tuned_distilbert_fake_news", batch_size=8):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probs.extend(probs[:, 1])
    return np.array(all_probs)


# Base Models 
base_models = [
    "Logistic_Regression",
    "Linear_SVC",
    "Multinomial_NB",
    "Random_Forest",
    "Gradient_Boosting"
]

# Generate Probabilities for all Models
train_meta_features = []
test_meta_features = []

print("\nGenerating base model probabilities...")

for model_name in base_models:
    print(f"→ {model_name}")
    train_probs = get_model_probs(X_train.tolist(), model_name)
    test_probs = get_model_probs(X_test.tolist(), model_name)
    train_meta_features.append(train_probs)
    test_meta_features.append(test_probs)

# DistilBERT
print("→ DistilBERT")
bert_train_probs = get_bert_probs(X_train.tolist())
bert_test_probs = get_bert_probs(X_test.tolist())
train_meta_features.append(bert_train_probs)
test_meta_features.append(bert_test_probs)

# Stack features horizontally
train_meta = np.vstack(train_meta_features).T
test_meta = np.vstack(test_meta_features).T

print(f"\nMeta feature shape: {train_meta.shape}")

# Train Meta-Classifier
print("\nTraining Meta-classifier (Logistic Regression)...")
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(train_meta, y_train)

# Predictions
meta_probs = meta_model.predict_proba(test_meta)[:, 1]
meta_preds = meta_model.predict(test_meta)

# Evaluate Meta-Classifier
accuracy = accuracy_score(y_test, meta_preds)
precision = precision_score(y_test, meta_preds)
recall = recall_score(y_test, meta_preds)
f1 = f1_score(y_test, meta_preds)
roc_auc = roc_auc_score(y_test, meta_probs)

print("\nMeta-classifier Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, meta_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Meta-classifier")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, meta_preds, target_names=["FAKE", "REAL"]))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, meta_probs)
plt.plot(fpr, tpr, label=f"Meta Model (AUC={roc_auc:.4f})", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Meta-classifier")
plt.legend()
plt.show()

# Save Meta-Model
joblib.dump(meta_model, "models/meta_classifier.pkl")
print("\nMeta-classifier saved as models/meta_model.pkl")

