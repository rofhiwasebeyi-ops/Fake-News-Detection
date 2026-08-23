# Detecting Disinformation: Fake News Classifier

A machine learning and natural language processing system for detecting fake news using classical machine learning, transformer-based models, and ensemble learning.

The project compares traditional **TF-IDF-based classifiers** with a fine-tuned **DistilBERT** model and combines model predictions using a **stacking ensemble**. The system also incorporates model calibration and explainability techniques to improve the reliability and interpretability of predictions.

---

## Overview

The rapid spread of misinformation and disinformation through online news platforms makes automated fake-news detection an important natural language processing task.

This project investigates different approaches to binary fake-news classification:

* Traditional machine learning using TF-IDF representations
* Transformer-based classification using DistilBERT
* Stacking ensemble learning
* Probability calibration using temperature scaling
* Model explainability using feature analysis and LIME
* Interactive prediction using a Streamlit application

The objective is to develop a reliable classifier capable of distinguishing between **fake** and **real** news articles.

---

## Features

* Text preprocessing and feature extraction
* TF-IDF-based feature representation
* Multiple classical machine learning classifiers
* Fine-tuned DistilBERT transformer model
* Stacking ensemble combining multiple models
* Probability calibration using temperature scaling
* Explainability for model predictions
* Interactive Streamlit web application
* Evaluation using multiple classification metrics
* Visualisation of model performance and predictions

---

## Dataset

The project combines multiple publicly available fake-news and misinformation datasets:

* **Fake.csv**
* **True.csv**
* **LIAR**
* **Misinformation**
* **FakeNewsNet**

The datasets were processed and combined to create a binary classification dataset containing **71,941 fake and 71,941 real news samples**.

The actual datasets are not included in this repository because of their size and distribution considerations.

Instructions and links for accessing the datasets are provided in:

```text
dataset_and_model_links.txt
```

---

## Methodology

The project follows a multi-stage classification pipeline.

```text
News Articles
     │
     ▼
Data Collection & Integration
     │
     ▼
Text Preprocessing
     │
     ├───────────────────────┐
     ▼                       ▼
TF-IDF Representation    DistilBERT
     │                       │
     ▼                       ▼
Classical ML Models      Fine-Tuning
     │                       │
     └───────────┬───────────┘
                 ▼
          Stacking Ensemble
                 │
                 ▼
       Probability Calibration
                 │
                 ▼
       Fake / Real Prediction
                 │
                 ▼
       Explainability Analysis
```

---

## Models

### Classical Machine Learning

The TF-IDF representation is used with several classical classifiers.

The models evaluated include:

* Logistic Regression
* Linear Support Vector Classifier (Linear SVC)
* Additional classifiers used as ensemble components

TF-IDF converts news articles into numerical feature vectors based on the importance of words and phrases within the corpus.

### DistilBERT

A pre-trained **DistilBERT** transformer model is fine-tuned for the fake-news classification task.

DistilBERT provides contextual representations of text and allows the system to capture relationships between words that traditional bag-of-words approaches may not represent effectively.

### Stacking Ensemble

A stacking ensemble combines predictions from multiple base models.

The ensemble uses the predictions of the individual models as inputs to a higher-level meta-model, allowing the system to learn how to combine the strengths of different classifiers.

The stacking ensemble achieved an accuracy of approximately:

**97.21%**

---

## Probability Calibration

Classification probabilities were calibrated using **temperature scaling**.

Calibration is important because a model can achieve high classification accuracy while producing probabilities that do not accurately represent its confidence.

Temperature scaling adjusts the model's output logits using a learned temperature parameter before converting them into probabilities.

This provides more reliable confidence estimates for predictions.

---

## Explainability

Model explainability techniques were incorporated to investigate why the classifiers make particular predictions.

### TF-IDF Feature Analysis

For linear TF-IDF models, feature coefficients can be inspected to identify words and terms that contribute strongly toward fake or real predictions.

### LIME

**Local Interpretable Model-Agnostic Explanations (LIME)** is used to provide local explanations for individual predictions, particularly for the transformer-based classifier.

LIME highlights parts of an input text that contribute to the model's prediction.

---

## Results

The models were evaluated using metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Selected Results

| Model                        |   Accuracy |
| ---------------------------- | ---------: |
| Logistic Regression + TF-IDF |     90.20% |
| Linear SVC + TF-IDF          |     93.73% |
| Stacking Ensemble            | **97.21%** |

The stacking ensemble provided the strongest overall performance among the evaluated approaches.

> Results may vary depending on preprocessing, training configuration, dataset splits, and model checkpoints.

---

## Project Structure

```text
Fake-News-Detection/
│
├── src/
│   ├── ...
│   └── ...
│
├── screenshots/
│   ├── ...
│   └── ...
│
├── README.md
├── requirements.txt
├── dataset_and_model_links.txt
└── .gitignore
```

### Directory Description

| Directory/File                | Description                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `src/`                        | Source code for preprocessing, training, evaluation, prediction, and application components |
| `screenshots/`                | Screenshots demonstrating the application and experimental results                          |
| `README.md`                   | Project documentation                                                                       |
| `requirements.txt`            | Python dependencies required to run the project                                             |
| `dataset_and_model_links.txt` | Links for accessing the datasets and trained model files                                    |
| `.gitignore`                  | Specifies files and directories that should not be committed                                |

The actual datasets and large trained model files are excluded from the repository.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rofhiwasebeyi-ops/Fake-News-Detection.git
```

```bash
cd Fake-News-Detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset and Model Setup

The datasets and trained model files are not stored directly in the GitHub repository.

Access information is provided in:

```text
dataset_and_model_links.txt
```

After obtaining the required files, place them in the appropriate directories according to the project configuration.

For example:

```text
Fake-News-Detection/
│
├── data/
│   └── dataset files
│
├── models/
│   └── trained model files
│
└── src/
```

The `data/` and `models/` directories are excluded from Git using `.gitignore`.

---

## Running the Project

The exact commands depend on which component of the project you want to run.

### Run the Python application

If the main application is located at:

```text
src/app.py
```

run:

```bash
python src/app.py
```

### Run the Streamlit application

If the Streamlit application is located at:

```text
src/app.py
```

run:

```bash
streamlit run src/app.py
```

This will launch the interactive fake-news detection application in your browser.

---

## Example Prediction Workflow

The application accepts news text as input and processes it through the trained classification pipeline.

```text
Input News Article
       │
       ▼
Text Preprocessing
       │
       ▼
Feature Extraction / Tokenisation
       │
       ▼
Trained Classification Model
       │
       ▼
Probability Calibration
       │
       ▼
Prediction
       │
       ├── REAL
       │
       └── FAKE
```

The application can also provide confidence information and model explanations where supported.

---

## Screenshots

Screenshots demonstrating the application, predictions, and experimental results are available in the:

```text
screenshots/
```

directory.

Example screenshots can include:

* Streamlit application interface
* Fake-news prediction
* Real-news prediction
* Model performance
* Confusion matrices
* Explainability results

---

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **TensorFlow / PyTorch**
* **Transformers**
* **DistilBERT**
* **NLTK**
* **LIME**
* **Matplotlib**
* **Streamlit**
* **Git / GitHub**

---

## Evaluation Metrics

The following metrics are used to evaluate model performance.

### Accuracy

Measures the proportion of correctly classified news articles.

### Precision

Measures how many articles predicted as fake are actually fake.

### Recall

Measures how many of the actual fake articles are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between fake and real news across different classification thresholds.

---

## Reproducibility

To reproduce the experiments:

1. Clone the repository.
2. Create a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Obtain the datasets using the links provided in `dataset_and_model_links.txt`.
5. Obtain the required trained models if inference rather than training is being performed.
6. Place the files in the required directories.
7. Run the appropriate scripts in `src/`.
8. Evaluate the resulting predictions using the provided evaluation functionality.

---

## Limitations

Although the system achieves strong classification performance, several limitations should be considered:

* Performance depends on the quality and distribution of the training datasets.
* Models may perform differently on news from sources or topics not represented in the training data.
* High classification accuracy does not guarantee that every individual prediction is correct.
* Transformer models can require significantly more computational resources than traditional machine learning models.
* Model explanations such as LIME provide approximations of model behaviour rather than guaranteed causal explanations.
* Dataset biases can be reflected in the resulting models.

---

## Future Improvements

Potential improvements include:

* Expanding the training data with additional recent news sources.
* Evaluating the models on completely unseen datasets.
* Improving cross-domain generalisation.
* Investigating additional transformer architectures.
* Exploring more advanced ensemble strategies.
* Improving probability calibration.
* Developing more robust explainability methods.
* Monitoring model performance on newly emerging misinformation.
* Deploying the classifier as a scalable web service.

---

## Repository

GitHub repository:

**https://github.com/rofhiwasebeyi-ops/Fake-News-Detection**

---

## Author

**Rofhiwa Sebeyi**

Computer Science Honours Student

---

## License

This project is intended for academic and educational purposes.

If a specific open-source license is required, a license should be added to the repository according to the licensing requirements of the project and any datasets or external resources used.
