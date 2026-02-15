import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, preprocess_pipeline, clean_text

def evaluate_model():
    print("Loading test data...")
    # Ideally should use a separate test set, but for simplicity here we reload the full dataset
    # and rely on the user understanding this is just a demo of loading artifacts.
    # In a real scenario, we'd save X_test/y_test during training or have a separate file.
    # For this script, let's just show how to load the model and predict on some sample data 
    # or the whole dataset again (biasing evaluation but proving pipeline works).
    
    # Better approach: Just do a manual test or load the full dataset and split it same way (random_state=42)
    # to get the test set again.
    
    df = load_data('data/spam.csv')
    df = preprocess_pipeline(df)
    
    print("Loading artifacts...")
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    # Re-create test set to evaluate properly
    from sklearn.model_selection import train_test_split
    X = vectorizer.transform(df['clean_text']).toarray()
    y = df['label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Predicting...")
    y_pred = model.predict(X_test)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../results_confusion_matrix.png')
    print("Confusion matrix saved to results_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
