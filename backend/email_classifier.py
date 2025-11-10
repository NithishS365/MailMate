"""
EmailClassifier module for MailMate - AI-driven email classification system.

This module provides the EmailClassifier class that supports:
- TF-IDF classification using scikit-learn
- Transformer-based classification using HuggingFace BERT/DistilBERT
- Training on labeled email text data
- Predicting email categories: Personal, Work, Finance, Promotions, Spam, Urgent
- Model persistence and loading
- Comprehensive evaluation metrics
"""

import json
import logging
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# HuggingFace imports (optional - will handle import errors gracefully)
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
    
    class EmailDataset(Dataset):
        """PyTorch Dataset for email classification with transformers."""
        
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("HuggingFace transformers not available. Only TF-IDF models will work.")
    EmailDataset = None

# Configure logging
logger = logging.getLogger(__name__)

# Email categories
EMAIL_CATEGORIES = ['Personal', 'Work', 'Finance', 'Promotions', 'Spam', 'Urgent']

@dataclass
class ClassificationResult:
    """Result of email classification."""
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float


class EmailClassifierError(Exception):
    """Base exception for EmailClassifier errors."""
    pass


class ModelNotTrainedError(EmailClassifierError):
    """Exception raised when trying to predict with untrained model."""
    pass


class EmailClassifier:
    """
    EmailClassifier for categorizing emails using multiple ML approaches.
    
    Supports:
    - TF-IDF with scikit-learn (LogisticRegression, RandomForest)
    - Transformer models (BERT, DistilBERT) via HuggingFace
    - Training on labeled data
    - Prediction with confidence scores
    - Model persistence and evaluation
    """
    
    def __init__(self, model_type: str = 'tfidf', classifier_type: str = 'logistic',
                 transformer_model: str = 'distilbert-base-uncased',
                 categories: Optional[List[str]] = None):
        """
        Initialize EmailClassifier.
        
        Args:
            model_type: 'tfidf' or 'transformer'
            classifier_type: For TF-IDF: 'logistic' or 'random_forest'
            transformer_model: HuggingFace model name for transformer approach
            categories: List of email categories (default: EMAIL_CATEGORIES)
        """
        self.model_type = model_type.lower()
        self.classifier_type = classifier_type.lower()
        self.transformer_model = transformer_model
        self.categories = categories or EMAIL_CATEGORIES.copy()
        
        # Model components
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.tokenizer = None
        self.pipeline_model = None
        
        # Training state
        self.is_trained = False
        self.training_history = {}
        
        # Validate inputs
        if self.model_type not in ['tfidf', 'transformer']:
            raise EmailClassifierError(f"Unsupported model_type: {model_type}")
        
        if self.model_type == 'transformer' and not TRANSFORMERS_AVAILABLE:
            raise EmailClassifierError("Transformers not available. Install with: pip install transformers torch")
        
        if self.classifier_type not in ['logistic', 'random_forest']:
            raise EmailClassifierError(f"Unsupported classifier_type: {classifier_type}")
        
        logger.info(f"Initialized EmailClassifier: {model_type} with {classifier_type}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess email text for classification.
        
        Args:
            text: Raw email text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     ' [URL] ', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' [EMAIL] ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length to prevent memory issues
        if len(text) > 10000:
            text = text[:10000]
        
        return text
    
    def prepare_training_data(self, texts: List[str], labels: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Prepare training data by preprocessing texts and encoding labels.
        
        Args:
            texts: List of email texts
            labels: List of email category labels
            
        Returns:
            Tuple of (preprocessed_texts, encoded_labels)
        """
        # Validate inputs
        if len(texts) != len(labels):
            raise EmailClassifierError("Number of texts and labels must match")
        
        if not texts or not labels:
            raise EmailClassifierError("Training data cannot be empty")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize and fit label encoder
        self.label_encoder = LabelEncoder()
        
        # Ensure all categories are represented
        all_categories = list(set(labels + self.categories))
        self.label_encoder.fit(all_categories)
        
        # Encode labels
        try:
            encoded_labels = self.label_encoder.transform(labels)
        except ValueError as e:
            unknown_labels = set(labels) - set(self.categories)
            if unknown_labels:
                logger.warning(f"Unknown labels found: {unknown_labels}. Adding to categories.")
                self.categories.extend(list(unknown_labels))
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.categories)
                encoded_labels = self.label_encoder.transform(labels)
            else:
                raise e
        
        logger.info(f"Prepared {len(processed_texts)} training samples with {len(self.categories)} categories")
        return processed_texts, encoded_labels
    
    def train_tfidf_model(self, texts: List[str], labels: np.ndarray,
                         max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2),
                         test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train TF-IDF based classifier.
        
        Args:
            texts: Preprocessed email texts
            labels: Encoded labels
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            test_size: Test set size for evaluation
            random_state: Random seed
            
        Returns:
            Training metrics
        """
        logger.info("Training TF-IDF model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Choose classifier
        if self.classifier_type == 'logistic':
            classifier = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.classifier_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', classifier)
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Classification report
        test_report = classification_report(
            y_test, test_predictions,
            target_names=[self.label_encoder.classes_[i] for i in range(len(self.label_encoder.classes_))],
            output_dict=True
        )
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': test_report,
            'feature_count': len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else max_features
        }
        
        self.is_trained = True
        self.training_history = metrics
        
        logger.info(f"TF-IDF model trained. Test accuracy: {test_accuracy:.4f}")
        return metrics
    
    def train_transformer_model(self, texts: List[str], labels: np.ndarray,
                               test_size: float = 0.2, batch_size: int = 16,
                               num_epochs: int = 3, learning_rate: float = 2e-5,
                               max_length: int = 512, random_state: int = 42) -> Dict[str, Any]:
        """
        Train transformer-based classifier.
        
        Args:
            texts: Preprocessed email texts
            labels: Encoded labels
            test_size: Test set size
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            max_length: Maximum sequence length
            random_state: Random seed
            
        Returns:
            Training metrics
        """
        if not TRANSFORMERS_AVAILABLE:
            raise EmailClassifierError("Transformers not available")
        
        logger.info(f"Training transformer model: {self.transformer_model}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.transformer_model,
            num_labels=len(self.categories)
        )
        
        # Create datasets
        train_dataset = EmailDataset(X_train, y_train.tolist(), self.tokenizer, max_length)
        test_dataset = EmailDataset(X_test, y_test.tolist(), self.tokenizer, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./email_classifier_results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train model
        training_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        # Create classification pipeline
        self.pipeline_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        
        # Test accuracy
        test_predictions = []
        for text in X_test:
            result = self.pipeline_model(text)[0]
            predicted_label = max(result, key=lambda x: x['score'])['label']
            # Convert LABEL_X to actual category
            predicted_idx = int(predicted_label.split('_')[1])
            test_predictions.append(predicted_idx)
        
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        metrics = {
            'train_loss': training_result.training_loss,
            'eval_loss': eval_result['eval_loss'],
            'test_accuracy': test_accuracy,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        self.is_trained = True
        self.training_history = metrics
        
        logger.info(f"Transformer model trained. Test accuracy: {test_accuracy:.4f}")
        return metrics
    
    def train(self, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """
        Train the classifier on labeled email data.
        
        Args:
            texts: List of email texts
            labels: List of email category labels
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics
        """
        # Prepare data
        processed_texts, encoded_labels = self.prepare_training_data(texts, labels)
        
        # Train based on model type
        if self.model_type == 'tfidf':
            return self.train_tfidf_model(processed_texts, encoded_labels, **kwargs)
        elif self.model_type == 'transformer':
            return self.train_transformer_model(processed_texts, encoded_labels, **kwargs)
        else:
            raise EmailClassifierError(f"Unknown model type: {self.model_type}")
    
    def predict(self, texts: Union[str, List[str]], 
                return_probabilities: bool = True) -> Union[ClassificationResult, List[ClassificationResult]]:
        """
        Predict email categories for given texts.
        
        Args:
            texts: Single email text or list of email texts
            return_probabilities: Whether to return probability scores
            
        Returns:
            ClassificationResult or list of ClassificationResults
        """
        import time
        
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before prediction")
        
        # Handle single text input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        results = []
        
        for text in texts:
            start_time = time.time()
            processed_text = self.preprocess_text(text)
            
            if self.model_type == 'tfidf':
                # TF-IDF prediction
                predicted_label_idx = self.model.predict([processed_text])[0]
                predicted_category = self.label_encoder.inverse_transform([predicted_label_idx])[0]
                
                if return_probabilities:
                    probabilities = self.model.predict_proba([processed_text])[0]
                    prob_dict = {
                        self.label_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(probabilities)
                    }
                    confidence = float(max(probabilities))
                else:
                    prob_dict = {}
                    confidence = 1.0
            
            elif self.model_type == 'transformer':
                # Transformer prediction
                prediction = self.pipeline_model(processed_text)[0]
                
                # Find the highest scoring prediction
                best_prediction = max(prediction, key=lambda x: x['score'])
                predicted_label_idx = int(best_prediction['label'].split('_')[1])
                predicted_category = self.categories[predicted_label_idx]
                confidence = float(best_prediction['score'])
                
                if return_probabilities:
                    prob_dict = {}
                    for pred in prediction:
                        label_idx = int(pred['label'].split('_')[1])
                        category = self.categories[label_idx]
                        prob_dict[category] = float(pred['score'])
                else:
                    prob_dict = {}
            
            processing_time = time.time() - start_time
            
            result = ClassificationResult(
                predicted_category=predicted_category,
                confidence=confidence,
                probabilities=prob_dict,
                processing_time=processing_time
            )
            
            results.append(result)
        
        return results[0] if single_input else results
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            texts: List of email texts
            true_labels: List of true category labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before evaluation")
        
        # Get predictions
        predictions = self.predict(texts, return_probabilities=False)
        predicted_labels = [pred.predicted_category for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels,
            target_names=self.categories,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=self.categories)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'categories': self.categories
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for TF-IDF models.
        
        Args:
            top_n: Number of top features per category
            
        Returns:
            Dictionary of category -> list of (feature, importance) tuples
        """
        if not self.is_trained or self.model_type != 'tfidf':
            raise EmailClassifierError("Feature importance only available for trained TF-IDF models")
        
        classifier = self.model.named_steps['classifier']
        feature_names = self.vectorizer.get_feature_names_out()
        
        importance_dict = {}
        
        if hasattr(classifier, 'coef_'):
            # For LogisticRegression
            for i, category in enumerate(self.label_encoder.classes_):
                if len(classifier.coef_.shape) == 1:
                    # Binary classification
                    coefficients = classifier.coef_
                else:
                    # Multi-class classification
                    coefficients = classifier.coef_[i]
                
                # Get top positive and negative features
                top_indices = np.argsort(np.abs(coefficients))[-top_n:]
                importance_dict[category] = [
                    (feature_names[idx], float(coefficients[idx]))
                    for idx in reversed(top_indices)
                ]
        
        elif hasattr(classifier, 'feature_importances_'):
            # For RandomForest
            importances = classifier.feature_importances_
            top_indices = np.argsort(importances)[-top_n:]
            
            # For tree-based models, we get overall importance
            importance_dict['overall'] = [
                (feature_names[idx], float(importances[idx]))
                for idx in reversed(top_indices)
            ]
        
        return importance_dict
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save trained model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")
        
        file_path = Path(file_path)
        
        model_data = {
            'model_type': self.model_type,
            'classifier_type': self.classifier_type,
            'transformer_model': self.transformer_model,
            'categories': self.categories,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        if self.model_type == 'tfidf':
            # Save TF-IDF model
            model_data['model'] = self.model
            model_data['label_encoder'] = self.label_encoder
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
        
        elif self.model_type == 'transformer':
            # Save transformer model
            model_dir = file_path.parent / f"{file_path.stem}_transformer"
            model_dir.mkdir(exist_ok=True)
            
            # Save the pipeline model components
            if self.pipeline_model:
                self.pipeline_model.save_pretrained(model_dir)
            
            # Save metadata
            with open(file_path, 'w') as f:
                json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: Union[str, Path]) -> None:
        """
        Load trained model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        if file_path.suffix == '.pkl':
            # Load TF-IDF model
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            
        else:
            # Load transformer model
            with open(file_path, 'r') as f:
                model_data = json.load(f)
            
            model_dir = file_path.parent / f"{file_path.stem}_transformer"
            
            if model_dir.exists() and TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                self.pipeline_model = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=self.tokenizer,
                    return_all_scores=True
                )
        
        # Restore metadata
        self.model_type = model_data['model_type']
        self.classifier_type = model_data['classifier_type']
        self.transformer_model = model_data['transformer_model']
        self.categories = model_data['categories']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        
        logger.info(f"Model loaded from {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        info = {
            'model_type': self.model_type,
            'classifier_type': self.classifier_type,
            'transformer_model': self.transformer_model,
            'categories': self.categories,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        if self.is_trained and self.model_type == 'tfidf' and self.vectorizer:
            try:
                info['vocabulary_size'] = len(self.vectorizer.get_feature_names_out())
            except:
                info['vocabulary_size'] = 'unknown'
        
        return info