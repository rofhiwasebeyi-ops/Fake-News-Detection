# Fake-News-Detection
A hybrid system to detect fake news articles using Machine Learning and Transformer-Based Model for Fake News Detection 

# Project Overview
This project is a **Machine Learning-based Fake News Detection System** that classifies news articles as **Real** or **Fake**.  
It is built to help combat misinformation by automatically analyzing news content using classical ML algorithms.

# Technologies Used
- **Programming Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
- **Algorithms Implemented:**  
  - Logistic Regression  
  - Linear SVM  
  - Multinomial Naive Bayes  
  - Random Forest  
  - Gradient Boosting  
- **Evaluation Metrics:** Accuracy, Confusion Matrix, ROC Curve

# How It Works
1. Load and preprocess the dataset (text cleaning, vectorization).  
2. Split data into training and testing sets.  
3. Train multiple machine learning models.  
4. Evaluate models using accuracy, confusion matrix, and ROC curve.  
5. Select the best-performing model for prediction.

# Results
- **Accuracy Example:** 93% (Logistic Regression)  
- Confusion Matrix and ROC Curve visuals.  
- Model selection is based on highest accuracy and best ROC-AUC score.

# Fine-Tuned DistilBERT Model
Download the trained model here:
  [Download Model][https://drive.google.com/drive/folders/1sbpzWzypRmq3nEM19zmmjrG5q1H0ifvf?usp=sharing]

Place it inside the 'models/fine_tuned_distilbert_fake_news/' folder before running the project.

# Dataset
Download the dataset here:
 [Download Model][https://drive.google.com/drive/folders/1C7EkZAzVELo4eq1vN2qt_HTvpjcZ4ZBH?usp=sharing]

Place it inside the 'data/' folder before running the project.
