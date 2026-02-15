import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1') # Handle potential encoding issues
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_text(text):
    """
    Clean and preprocess text:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    - Stemming
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization (simple split by whitespace)
    tokens = text.split()
    
    # Remove stopwords and stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(clean_tokens)

def preprocess_pipeline(df, text_column='message'):
    """
    Apply cleaning to the text column.
    """
    print("Preprocessing text data...")
    df['clean_text'] = df[text_column].apply(clean_text)
    return df

if __name__ == "__main__":
    # Test execution
    sample_text = "Check out this FREE offer! Call now."
    print(f"Original: {sample_text}")
    print(f"Cleaned: {clean_text(sample_text)}")
