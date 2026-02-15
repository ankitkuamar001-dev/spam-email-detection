# Spam Email Detection using NLP

## Project Overview
This project builds a machine learning model to classify SMS messages as 'spam' or 'ham' (not spam). It uses TF-IDF for feature extraction and compares Naive Bayes, Logistic Regression, and Random Forest classifiers. The best performing model (Random Forest) is deployed via a Streamlit web application.

## Dataset
- **Name**: SMS Spam Collection
- **Source**: UCI Machine Learning Repository
- **Size**: 5572 messages

## Project Structure
```
spam-email-detection/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks (optional)
├── src/                # Source code
│   ├── data_setup.py   # Convert raw data to CSV
│   ├── preprocessing.py# Text cleaning pipeline
│   ├── train.py        # Model training and artifact saving
│   └── evaluate.py     # Evaluation metrics and plots
├── models/             # Saved model artifacts
├── app/                # Streamlit application
│   └── streamlit_app.py
├── requirements.txt    # Project dependencies
└── README.md
```

## Setup and Installation

1. **Clone/Navigate to the directory**:
   ```bash
   cd spam-email-detection
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Setup**:
   The dataset is downloaded automatically. To convert it to CSV:
   ```bash
   python src/data_setup.py
   ```

## Usage

### Train the Model
To retrain the models and save the best one:
```bash
python src/train.py
```
*Outputs `models/model.pkl` and `models/vectorizer.pkl`.*

### Evaluate the Model
To view classification metrics and generate a confusion matrix:
```bash
python src/evaluate.py
```

### Run the Web App
To launch the interactive spam detector:
```bash
streamlit run app/streamlit_app.py
```

## Results
- **Best Model**: Random Forest
- **Accuracy**: ~97%
- **Spam Recall**: ~83% (Ability to catch spam)

## Future Improvements
- Hyperparameter tuning using GridSearchCV.
- Deep Learning approaches (LSTM/BERT).
- Handling class imbalance with SMOTE.
