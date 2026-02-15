import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_data, preprocess_pipeline

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def train_models():
    # Load data
    print("Loading data...")
    df = load_data('data/spam.csv')
    if df is None:
        return

    # Preprocess
    df = preprocess_pipeline(df, text_column='message')
    
    # Feature Engineering
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    # Train and Evaluate
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # Save Artifacts
    print("Saving artifacts...")
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save label encoder/decoder if needed, but for now simple string labels work with these models
    # (sklearn handles string labels for target y usually)

if __name__ == "__main__":
    train_models()
