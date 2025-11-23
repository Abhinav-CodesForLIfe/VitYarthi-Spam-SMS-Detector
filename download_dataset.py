import pandas as pd
import requests
import os
from zipfile import ZipFile

def download_spam_dataset():
    """
    Download the SMS Spam Collection dataset
    """
    print("Downloading SMS Spam Collection dataset...")
    
    # URL for the SMS Spam Collection dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the zip file
        zip_path = 'data/smsspamcollection.zip'
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract the zip file
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/')
        
        # Read and display the dataset
        df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
        print(f"Dataset downloaded successfully! Shape: {df.shape}")
        print(f"Spam messages: {df[df['label'] == 'spam'].shape[0]}")
        print(f"Ham messages: {df[df['label'] == 'ham'].shape[0]}")
        
        # Clean up zip file
        os.remove(zip_path)
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def create_sample_dataset():
    """
    Create a sample dataset if download fails
    """
    print("Creating sample dataset...")
    
    sample_data = {
        'label': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
        'message': [
            'Hey, how are you doing today?',
            'Remember to buy milk on your way home',
            'WINNER!! You have won a $1000 gift card! Call now to claim.',
            'Lets meet for lunch tomorrow at 1 PM',
            'URGENT: Your bank account needs verification. Click here now!',
            'The meeting is scheduled for 3 PM today',
            'FREE entry to win iPhone 15! Reply YES to participate',
            'Can you send me the report by EOD?'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_spam_data.csv', index=False)
    print("Sample dataset created!")
    return df

if __name__ == "__main__":
    df = download_spam_dataset()
    if df is None:
        df = create_sample_dataset()
    
    print("\nFirst 5 messages:")
    print(df.head())