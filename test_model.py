import pandas as pd
import joblib
from spam_detector import SpamDetector
from preprocess_data import download_nltk_data

def test_sample_messages():
    """Test the model with sample messages"""
    download_nltk_data()
    detector = SpamDetector()
    
    if detector.model is None:
        return
    
    # Sample messages for testing
    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "WINNER!! You've been selected for a $1000 Walmart gift card. Text YES to claim!",
        "Your package will be delivered today between 2-4 PM",
        "URGENT: Your bank account needs verification. Click here: http://fake-bank.com",
        "Reminder: Doctor's appointment at 3 PM today",
        "Congratulations! You won an iPhone 15. Call now to claim your prize!",
        "Can you pick up the kids from school today?",
        "FREE entry to win a luxury car! Reply STOP to unsubscribe"
    ]
    
    print("Testing Spam Detector with Sample Messages")
    print("=" * 60)
    
    results = detector.predict_batch(test_messages)
    
    for result in results:
        print(f"\nMessage: {result['message']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Spam Probability: {result['spam_probability']:.4f}")
        print(f"Confidence: {'High' if abs(result['spam_probability'] - 0.5) > 0.3 else 'Medium'}")
        print("-" * 40)

def test_accuracy():
    """Test model accuracy on the test set"""
    try:
        # Load test data
        df = pd.read_csv('data/processed_spam_data.csv')
        
        # Load model
        detector = SpamDetector()
        if detector.model is None:
            return
        
        # Test on entire dataset
        correct = 0
        total = len(df)
        
        print("Testing model accuracy on entire dataset...")
        print("=" * 50)
        
        for idx, row in df.iterrows():
            result = detector.predict(row['message'])
            actual = 'SPAM' if row['label_binary'] == 1 else 'HAM'
            predicted = result['prediction']
            
            if actual == predicted:
                correct += 1
        
        accuracy = correct / total
        print(f"Accuracy on entire dataset: {accuracy:.4f} ({correct}/{total})")
        
    except Exception as e:
        print(f"Error testing accuracy: {e}")

if __name__ == "__main__":
    print("SPAM DETECTOR TESTING")
    print("=" * 50)
    
    test_sample_messages()
    print("\n" + "=" * 50)
    test_accuracy()