Spam SMS Detection System
A comprehensive machine learning solution for detecting spam messages using Python and various classification algorithms.

📋 Table of Contents
Overview

Features

Installation

Usage

Project Structure

Models

Results

Dataset

Contributing

License

🎯 Overview
The Spam SMS Detection System is a machine learning-based solution that automatically classifies text messages as spam or legitimate (ham). This project implements multiple classification algorithms with natural language processing techniques to achieve high accuracy in spam detection.

Key Highlights:

97.85% accuracy with Logistic Regression

Multiple algorithm comparison

Real-time message classification

Comprehensive text preprocessing

Easy-to-use command-line interface

✨ Features
Multiple ML Algorithms: Logistic Regression, Naive Bayes, Random Forest, SVM

Advanced Text Processing: TF-IDF vectorization with n-grams

Comprehensive Evaluation: Cross-validation and performance metrics

Interactive Detection: Real-time spam classification

Model Persistence: Save and load trained models

Batch Processing: Classify multiple messages at once

🚀 Installation
Prerequisites
Python 3.7 or higher

pip package manager

Step-by-Step Setup
Clone the repository

bash
git clone https://github.com/yourusername/spam-detection-ml.git
cd spam-detection-ml
Create virtual environment

bash
# Windows
python -m venv spam_env
spam_env\Scripts\activate

# Linux/Mac
python -m venv spam_env
source spam_env/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Download NLTK data (automatically handled by the scripts)

📁 Project Structure
text
spam-detection-ml/
├── data/                 # Dataset directory
│   ├── SMSSpamCollection    # Original dataset
│   └── processed_spam_data.csv  # Processed data
├── models/              # Trained models directory
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── *.pkl (other models)
├── download_dataset.py  # Dataset download script
├── preprocess_data.py   # Text preprocessing
├── train_model.py       # Model training
├── spam_detector.py     # Interactive detection
├── test_model.py        # Model testing
├── requirements.txt     # Dependencies
└── README.md           # Project documentation
🛠️ Usage
Complete Pipeline
Run the entire workflow in sequence:

bash
# 1. Download dataset
python download_dataset.py

# 2. Preprocess data
python preprocess_data.py

# 3. Train models
python train_model.py

# 4. Test the system
python test_model.py

# 5. Use interactively
python spam_detector.py
Individual Components
Download Dataset

bash
python download_dataset.py
Downloads the SMS Spam Collection dataset from UCI repository.

Preprocess Data

bash
python preprocess_data.py
Cleans and preprocesses text data (tokenization, stemming, etc.).

Train Models

bash
python train_model.py
Trains multiple ML algorithms and selects the best performer.

Interactive Detection

bash
python spam_detector.py
Launches interactive mode for real-time spam classification.

Batch Testing

bash
python test_model.py
Tests the model with sample messages and shows performance metrics.

🤖 Models Implemented
The system trains and compares four machine learning algorithms:

Logistic Regression - Best performer (97.85% accuracy)

Support Vector Machine (SVM) - 97.67% accuracy

Random Forest - 97.40% accuracy

Naive Bayes - 96.86% accuracy

Model Selection
The system automatically selects the best model based on cross-validation accuracy and saves it for future use.

📊 Results
Performance Metrics
Overall Accuracy: 97.85%

Precision (Spam): 99%

Recall (Spam): 87%

F1-Score (Spam): 93%

Classification Report
text
              precision    recall  f1-score   support

         Ham       0.98      1.00      0.99       965
        Spam       0.99      0.87      0.93       150

    accuracy                           0.98      1115
   macro avg       0.99      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115
Example Predictions
text
Input: "WINNER!! You won $1000 gift card!"
Output: SPAM (98.7% confidence)

Input: "Hey, let's meet for lunch tomorrow"
Output: HAM (99.2% confidence)
📈 Dataset
The project uses the SMS Spam Collection Dataset from UCI Machine Learning Repository.

Dataset Statistics:

Total Messages: 5,572

Spam Messages: 747 (13.4%)

Ham Messages: 4,825 (86.6%)

Format: TSV file with label and message columns

Sample Data:

text
ham     Hey, how are you doing?
spam    FREE entry to win iPhone! Reply YES
ham     Meeting at 3 PM today
spam    Urgent: Your account needs verification
🔧 Technical Details
Text Preprocessing Pipeline
Text Cleaning: Remove URLs, emails, special characters

Lowercasing: Convert all text to lowercase

Tokenization: Split text into individual words

Stemming: Reduce words to their root form

Stopword Removal: Remove common English words

Feature Engineering
TF-IDF Vectorization: Term Frequency-Inverse Document Frequency

N-grams: Unigrams and bigrams (1,2)

Max Features: 5,000 most important features

Model Training
Train-Test Split: 80-20 stratified split

Cross-Validation: 5-fold cross-validation

Hyperparameters: Default scikit-learn parameters

🐛 Troubleshooting
Common Issues
NLTK Data Download Error

bash
# Manual NLTK download
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
Virtual Environment Issues

bash
# Windows PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\spam_env\Scripts\Activate.ps1
Missing Dependencies

bash
pip install --upgrade pip
pip install -r requirements.txt
🤝 Contributing
We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.

Development Setup
Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request

Guidelines
Follow PEP 8 coding standards

Add docstrings for new functions

Update documentation as needed

Test your changes thoroughly

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
UCI Machine Learning Repository for the SMS Spam Collection dataset

Scikit-learn team for excellent machine learning libraries

NLTK team for natural language processing tools

📞 Support
If you encounter any problems or have questions:

Check existing issues on GitHub

Create a new issue with detailed description

Email: abhinav.25bai11330@vitbhopal.ac.in

Documentation: See code comments and docstrings

🔮 Future Enhancements
Planned features and improvements:

Web interface using Flask/Django

REST API for integration

Deep learning models (LSTM, BERT)

Multi-language support

Real-time model retraining

Mobile application

Cloud deployment

Ensemble methods
