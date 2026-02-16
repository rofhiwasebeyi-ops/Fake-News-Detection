import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import joblib
import warnings
warnings.filterwarnings("ignore")


# Helper Functions
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc


# Load Dataset
df = pd.read_csv("data/preprocessed/preprocessed_dataset.csv", na_filter=False)
df = df[df['clean_content'].notna() & (df['clean_content'].str.strip() != '')]

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(max_df=0.7, min_df=3, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Class Weights
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights_array))
print(f"Computed class weights: {class_weights}")

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Model Definitions
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Linear SVC": LinearSVC(max_iter=5000),
    "Multinomial NB": MultinomialNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Hyperparameter Grids
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "saga"],
        "class_weight": ['balanced', None]
    },
    "Linear SVC": {
        "C": [0.01, 0.1, 1, 10, 100],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "class_weight": ['balanced', None]
    },
    "Multinomial NB": {
        "alpha": [0.1, 0.5, 1.0]
    },
    "Random Forest": {
        "n_estimators": [50, 100],       
        "max_depth": [10, 20],         
        "min_samples_split": [5, 10],    
        "class_weight": ['balanced']
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100],        
        "learning_rate": [0.05, 0.1],     
        "max_depth": [3, 5],              
        "subsample": [0.8]               
    }
}

# Train, Tune, Evaluate, and Save
results = {}

for name, model in models.items():
    print(f"\nTuning and training {name}...")
    grid = param_grids.get(name)
    best_model = model  

    try:
        if grid:
            gs = GridSearchCV(
                estimator=model,
                param_grid=grid,
                scoring='accuracy',
                cv=3,            
                n_jobs=1,         
                verbose=1
            )
            if name in ["Gradient Boosting", "Multinomial NB"]:
                gs.fit(X_train_vec, y_train, sample_weight=sample_weights)
            else:
                gs.fit(X_train_vec, y_train)
            best_model = gs.best_estimator_
            print(f"Best params for {name}: {gs.best_params_}")
        else:
            if name in ["Gradient Boosting", "Multinomial NB"]:
                model.fit(X_train_vec, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = best_model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, name)

        # ROC Curve
        if hasattr(best_model, "predict_proba"):
            y_scores = best_model.predict_proba(X_test_vec)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_scores = best_model.decision_function(X_test_vec)
        else:
            print(f"{name} does not support probability estimates, skipping ROC curve.")
            y_scores = None

        if y_scores is not None:
            roc_auc = plot_roc_curve(y_test, y_scores, name)
            print(f"ROC AUC score for {name}: {roc_auc:.4f}")

        # Save tuned model
        model_path = f"models/{name.replace(' ', '_')}_tuned_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"{name} saved successfully at {model_path}")

    except Exception as e:
        print(f"Skipping {name} due to error: {e}")
        continue


# Accuracy Comparison
plt.figure(figsize=(10, 6))
bars = []
models_list = list(results.keys())
accuracies = list(results.values())
best_index = accuracies.index(max(accuracies))

for i, acc in enumerate(accuracies):
    color = 'green' if i == best_index else 'skyblue'
    bars.append(plt.bar(models_list[i], acc, color=color))

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")

for i, bar in enumerate(bars):
    height = bar[0].get_height()
    plt.text(bar[0].get_x() + bar[0].get_width()/2, height - 0.05, f'{height:.2f}', 
             ha='center', color='black', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\nBest Model: {models_list[best_index]} with accuracy {accuracies[best_index]:.2f}")

# Save Vectorizer
joblib.dump(vectorizer, "models/tf-idf_vectorizer.pkl")
print("TF-IDF Vectorizer saved successfully!")