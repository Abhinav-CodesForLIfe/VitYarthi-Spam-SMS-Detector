import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import os

# Download required NLTK data
def download_nltk_data():
    """Download all required NLTK datasets"""
    try:
        print("Downloading NLTK data...")
        
        # Download all required NLTK packages
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        nltk.download('punkt_tab', quiet=False)  # This is the missing resource
        nltk.download('averaged_perceptron_tagger', quiet=False)
        
        print("NLTK data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        # Try alternative download method
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
            print("NLTK data downloaded successfully with SSL workaround!")
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            print("Please check your internet connection and try again.")

class TextPreprocessor:
    def __init__(self):
        download_nltk_data()  # Ensure NLTK data is downloaded
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and numbers - keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """Tokenize text and apply stemming"""
        try:
            # If text is empty after cleaning, return empty string
            if not text.strip():
                return ""
            
            tokens = word_tokenize(text)
            filtered_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if token not in self.stop_words and token not in string.punctuation
            ]
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return text  # Return original text if tokenization fails
    
    def preprocess_dataframe(self, df, text_column='message'):
        """Preprocess entire dataframe"""
        print("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        print("Tokenizing and stemming...")
        # Show progress for large datasets
        if len(df) > 1000:
            print("This may take a while for large datasets...")
        
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_stem)
        
        # Convert labels to binary (spam=1, ham=0)
        df['label_binary'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Check for any NaN values in processed_text
        nan_count = df['processed_text'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in processed_text. Filling with empty strings.")
            df['processed_text'] = df['processed_text'].fillna('')
        
        print("Preprocessing completed!")
        print(f"Dataset shape: {df.shape}")
        print(f"Sample processed text: {df['processed_text'].iloc[0][:100]}...")
        
        return df

def load_data():
    """Load the dataset"""
    try:
        # Try to load the real dataset first
        dataset_path = os.path.join('data', 'SMSSpamCollection')
        df = pd.read_csv(dataset_path, sep='\t', header=None, names=['label', 'message'])
        print(f"Real dataset loaded: {df.shape[0]} messages")
    except FileNotFoundError:
        try:
            # Try sample dataset
            sample_path = os.path.join('data', 'sample_spam_data.csv')
            df = pd.read_csv(sample_path)
            print(f"Sample dataset loaded: {df.shape[0]} messages")
        except FileNotFoundError:
            print("No dataset found. Please run download_dataset.py first.")
            return None
    
    print(f"Spam: {df[df['label'] == 'spam'].shape[0]}, Ham: {df[df['label'] == 'ham'].shape[0]}")
    return df

# Alternative simple preprocessing without NLTK
def simple_preprocess_text(text):
    """Simple preprocessing without NLTK dependencies"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple word splitting (alternative to tokenization)
    words = text.split()
    
    # Simple stop words removal
    simple_stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    filtered_words = [word for word in words if word not in simple_stop_words and len(word) > 2]
    
    return ' '.join(filtered_words)

def simple_preprocess(df, text_column='message'):
    """Simple preprocessing without advanced NLP"""
    print("Using simple preprocessing (no NLTK required)...")
    
    df['processed_text'] = df[text_column].apply(simple_preprocess_text)
    df['label_binary'] = df['label'].map({'spam': 1, 'ham': 0})
    
    return df

if __name__ == "__main__":
    # Load data
    df = load_data()
    if df is not None:
        try:
            # Try advanced preprocessing with NLTK
            preprocessor = TextPreprocessor()
            df_processed = preprocessor.preprocess_dataframe(df)
        except Exception as e:
            print(f"Advanced preprocessing failed: {e}")
            print("Falling back to simple preprocessing...")
            # Fall back to simple preprocessing
            df_processed = simple_preprocess(df)
        
        # Save processed data
        processed_path = os.path.join('data', 'processed_spam_data.csv')
        df_processed.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")
        
        print("\nSample of processed data:")
        print(df_processed[['label', 'message', 'processed_text']].head())
        
        # Show data info
        print(f"\nFinal dataset info:")
        print(f"Total messages: {len(df_processed)}")
        print(f"Spam: {df_processed['label_binary'].sum()}")
        print(f"Ham: {len(df_processed) - df_processed['label_binary'].sum()}")