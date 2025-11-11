import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from typing import List, Dict, Tuple

class EmailClassifier:
    """
    Machine Learning classifier for email categorization and prioritization
    Uses both Naive Bayes and Logistic Regression
    """
    
    def __init__(self, model_type: str = "naive_bayes"):
        """
        Initialize the classifier
        
        Args:
            model_type: Either "naive_bayes" or "logistic_regression"
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        
        if model_type == "naive_bayes":
            self.category_model = MultinomialNB(alpha=1.0)
            self.priority_model = MultinomialNB(alpha=1.0)
        else:  # logistic_regression
            self.category_model = LogisticRegression(max_iter=1000, random_state=42)
            self.priority_model = LogisticRegression(max_iter=1000, random_state=42)
        
        self.is_trained = False
    
    def prepare_features(self, emails_df: pd.DataFrame) -> np.ndarray:
        """Prepare features from email DataFrame"""
        # Combine subject and body for text analysis
        if 'full_content_cleaned' not in emails_df.columns:
            emails_df['full_content_cleaned'] = (
                emails_df['subject'].fillna('') + ' ' + emails_df['body'].fillna('')
            )
        
        return emails_df['full_content_cleaned'].fillna('').values
    
    def train(self, emails_df: pd.DataFrame) -> Dict:
        """
        Train the classifier on email data
        
        Args:
            emails_df: DataFrame containing emails with 'category' labels
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X_text = self.prepare_features(emails_df)
        
        # Transform text to TF-IDF features
        X = self.vectorizer.fit_transform(X_text)
        
        # Encode category labels
        y_category = self.category_encoder.fit_transform(emails_df['category'].values)
        
        # Assign priorities based on categories
        priority_map = {
            'urgent': 'high',
            'work': 'medium',
            'personal': 'low',
            'promotion': 'low',
            'spam': 'low'
        }
        emails_df['priority'] = emails_df['category'].map(priority_map)
        y_priority = self.priority_encoder.fit_transform(emails_df['priority'].values)
        
        # Split data for validation
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X, y_category, test_size=0.2, random_state=42
        )
        
        _, _, y_pri_train, y_pri_test = train_test_split(
            X, y_priority, test_size=0.2, random_state=42
        )
        
        # Train category model
        self.category_model.fit(X_train, y_cat_train)
        y_cat_pred = self.category_model.predict(X_test)
        cat_accuracy = accuracy_score(y_cat_test, y_cat_pred)
        
        # Train priority model
        self.priority_model.fit(X_train, y_pri_train)
        y_pri_pred = self.priority_model.predict(X_test)
        pri_accuracy = accuracy_score(y_pri_test, y_pri_pred)
        
        self.is_trained = True
        
        return {
            "category_accuracy": float(cat_accuracy),
            "priority_accuracy": float(pri_accuracy),
            "categories": self.category_encoder.classes_.tolist(),
            "priorities": self.priority_encoder.classes_.tolist(),
            "model_type": self.model_type,
            "training_samples": X_train.shape[0]  # Use shape[0] for sparse matrix
        }
    
    def predict_category(self, email_text: str) -> Tuple[str, float]:
        """
        Predict category for a single email
        
        Args:
            email_text: Combined subject and body text
            
        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.vectorizer.transform([email_text])
        prediction = self.category_model.predict(X)[0]
        probabilities = self.category_model.predict_proba(X)[0]
        
        category = self.category_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        return category, confidence
    
    def predict_priority(self, email_text: str) -> Tuple[str, float]:
        """
        Predict priority for a single email
        
        Args:
            email_text: Combined subject and body text
            
        Returns:
            Tuple of (predicted_priority, priority_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.vectorizer.transform([email_text])
        prediction = self.priority_model.predict(X)[0]
        probabilities = self.priority_model.predict_proba(X)[0]
        
        priority = self.priority_encoder.inverse_transform([prediction])[0]
        priority_score = float(max(probabilities))
        
        return priority, priority_score
    
    def predict_batch(self, emails_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict categories and priorities for multiple emails
        
        Args:
            emails_df: DataFrame containing emails
            
        Returns:
            DataFrame with predictions added
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_text = self.prepare_features(emails_df)
        X = self.vectorizer.transform(X_text)
        
        # Predict categories
        cat_predictions = self.category_model.predict(X)
        cat_probabilities = self.category_model.predict_proba(X)
        
        emails_df['predicted_category'] = self.category_encoder.inverse_transform(cat_predictions)
        emails_df['category_confidence'] = cat_probabilities.max(axis=1)
        
        # Predict priorities
        pri_predictions = self.priority_model.predict(X)
        pri_probabilities = self.priority_model.predict_proba(X)
        
        emails_df['predicted_priority'] = self.priority_encoder.inverse_transform(pri_predictions)
        emails_df['priority_score'] = pri_probabilities.max(axis=1)
        
        return emails_df
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'vectorizer': self.vectorizer,
            'category_model': self.category_model,
            'priority_model': self.priority_model,
            'category_encoder': self.category_encoder,
            'priority_encoder': self.priority_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.category_model = model_data['category_model']
        self.priority_model = model_data['priority_model']
        self.category_encoder = model_data['category_encoder']
        self.priority_encoder = model_data['priority_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
    
    def get_top_priority_emails(self, emails_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Get top N priority emails based on priority scores
        
        Args:
            emails_df: DataFrame with predicted priorities
            top_n: Number of top emails to return
            
        Returns:
            DataFrame with top priority emails
        """
        return emails_df.nlargest(top_n, 'priority_score')
