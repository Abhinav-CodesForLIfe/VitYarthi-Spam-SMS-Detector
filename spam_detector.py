import joblib
import pandas as pd
import numpy as np
from preprocess_data import TextPreprocessor, download_nltk_data

class SpamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            # Load best model
            best_model_info = joblib.load('models/best_model.pkl')
            self.model = best_model_info['model']
            self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            print(f"Model loaded: {best_model_info['name']}")
        except FileNotFoundError:
            print("Trained model not found. Please run train_model.py first.")
            self.model = None
            self.vectorizer = None
    
    def predict(self, message):
        """Predict if a message is spam"""
        if self.model is None or self.vectorizer is None:
            return "Error: Model not loaded. Please train the model first."
        
        # Preprocess the message
        cleaned_text = self.preprocessor.clean_text(message)
        processed_text = self.preprocessor.tokenize_and_stem(cleaned_text)
        
        # Vectorize
        message_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(message_vector)[0]
        probability = self.model.predict_proba(message_vector)[0]
        
        result = {
            'message': message,
            'is_spam': bool(prediction),
            'spam_probability': probability[1],
            'ham_probability': probability[0],
            'prediction': 'SPAM' if prediction else 'HAM'
        }
        
        return result
    
    def predict_batch(self, messages):
        """Predict multiple messages"""
        results = []
        for message in messages:
            results.append(self.predict(message))
        return results

def main():
    # Download NLTK data if needed
    download_nltk_data()
    
    # Initialize detector
    detector = SpamDetector()
    
    if detector.model is None:
        return
    
    # Interactive prediction loop
    print("\n" + "="*50)
    print("SPAM DETECTOR - Interactive Mode")
    print("="*50)
    print("Type 'quit' to exit\n")
    
    while True:
        message = input("Enter a message to check: ").strip()
        
        if message.lower() == 'quit':
            break
        
        if not message:
            continue
        
        result = detector.predict(message)
        
        print(f"\nResult: {result['prediction']}")
        print(f"Spam Probability: {result['spam_probability']:.4f}")
        print(f"Ham Probability: {result['ham_probability']:.4f}")
        
        if result['is_spam']:
            print("🚫 This message appears to be SPAM!")
        else:
            print("✅ This message appears to be HAM (not spam)")
        
        print("-" * 50)

if __name__ == "__main__":
    main()