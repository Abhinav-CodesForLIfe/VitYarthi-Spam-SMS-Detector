import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load processed data and handle missing values"""
        try:
            df = pd.read_csv('data/processed_spam_data.csv')
            
            # Check for missing values
            print(f"Dataset shape: {df.shape}")
            print(f"Missing values in processed_text: {df['processed_text'].isna().sum()}")
            print(f"Missing values in label_binary: {df['label_binary'].isna().sum()}")
            
            # Handle missing values
            if df['processed_text'].isna().sum() > 0:
                print("Filling missing processed_text with empty strings...")
                df['processed_text'] = df['processed_text'].fillna('')
            
            if df['label_binary'].isna().sum() > 0:
                print("Removing rows with missing labels...")
                df = df.dropna(subset=['label_binary'])
            
            # Check for empty processed texts
            empty_texts = (df['processed_text'].str.strip() == '').sum()
            print(f"Empty processed texts: {empty_texts}")
            
            return df
        except FileNotFoundError:
            print("Processed data not found. Please run preprocess_data.py first.")
            return None
    
    def prepare_features(self, df):
        """Prepare features and labels with data validation"""
        print("Preparing features...")
        
        # Ensure we have valid text data
        valid_mask = df['processed_text'].notna() & (df['processed_text'].str.strip() != '')
        print(f"Valid texts: {valid_mask.sum()}/{len(df)}")
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid text data found for training!")
        
        # Use only valid data
        df_valid = df[valid_mask].copy()
        
        X = self.vectorizer.fit_transform(df_valid['processed_text'])
        y = df_valid['label_binary']
        
        print(f"Feature matrix shape: {X.shape}")
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train all models and select the best one"""
        best_score = 0
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                mean_cv_score = np.mean(cv_scores)
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                results[name] = {
                    'model': model,
                    'cv_score': mean_cv_score
                }
                
                print(f"{name} - Cross-val Accuracy: {mean_cv_score:.4f}")
                
                # Update best model
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not results:
            raise ValueError("No models were successfully trained!")
            
        print(f"\nBest model: {self.best_model_name} with CV accuracy: {best_score:.4f}")
        return results
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        plt.savefig(f'models/confusion_matrix_{model_name.replace(" ", "_")}.png')
        plt.close()
        
        return accuracy
    
    def save_models(self, results):
        """Save all trained models and vectorizer"""
        os.makedirs('models', exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        print("Saved TF-IDF vectorizer")
        
        # Save models
        for name, result in results.items():
            filename = f'models/{name.replace(" ", "_").lower()}_model.pkl'
            joblib.dump(result['model'], filename)
            print(f"Saved {name} to {filename}")
        
        # Save best model info
        best_model_info = {
            'name': self.best_model_name,
            'model': self.best_model
        }
        joblib.dump(best_model_info, 'models/best_model.pkl')
        print(f"\nBest model ({self.best_model_name}) saved!")

def check_data_quality(df):
    """Check the quality of the dataset"""
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    print(f"Total samples: {len(df)}")
    print(f"Spam samples: {df['label_binary'].sum()}")
    print(f"Ham samples: {len(df) - df['label_binary'].sum()}")
    print(f"Missing processed_text: {df['processed_text'].isna().sum()}")
    print(f"Empty processed_text: {(df['processed_text'].str.strip() == '').sum()}")
    
    # Sample some processed texts
    print("\nSample processed texts:")
    for i in range(min(3, len(df))):
        text = df['processed_text'].iloc[i]
        print(f"  {i+1}. '{text[:50]}...'")
    
    print("="*50 + "\n")

def main():
    # Initialize classifier
    classifier = SpamClassifier()
    
    # Load data
    df = classifier.load_data()
    if df is None:
        print("Please run preprocess_data.py first to create the dataset.")
        return
    
    # Check data quality
    check_data_quality(df)
    
    # Prepare features
    try:
        X, y = classifier.prepare_features(df)
    except Exception as e:
        print(f"Error preparing features: {e}")
        print("The processed data might be corrupted. Please run preprocess_data.py again.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    try:
        results = classifier.train_models(X_train, y_train)
    except Exception as e:
        print(f"Error training models: {e}")
        return
    
    # Evaluate all models on test set
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    test_accuracies = {}
    for name, result in results.items():
        accuracy = classifier.evaluate_model(result['model'], X_test, y_test, name)
        test_accuracies[name] = accuracy
    
    # Save models
    classifier.save_models(results)
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    for name, accuracy in test_accuracies.items():
        print(f"{name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()