# EmailClassifier Documentation

## Overview

The `EmailClassifier` class is a sophisticated Python implementation for MailMate's AI-driven email classification system. It supports multiple machine learning approaches including traditional TF-IDF with scikit-learn and state-of-the-art Transformer models using HuggingFace, enabling automatic categorization of emails into predefined categories.

## Features

### 1. **Multiple Model Types**

- **TF-IDF + Scikit-learn**: Fast, lightweight classification using traditional ML
  - Logistic Regression
  - Random Forest
- **Transformer Models**: State-of-the-art NLP using HuggingFace
  - BERT, DistilBERT, RoBERTa
  - Fine-tuned for email classification

### 2. **Email Categories**

Pre-defined categories optimized for email management:

- **Personal**: Family, friends, personal communications
- **Work**: Business, meetings, project communications
- **Finance**: Banking, investments, financial statements
- **Promotions**: Marketing, sales, promotional content
- **Spam**: Unwanted, suspicious, or malicious emails
- **Urgent**: Time-sensitive, critical communications

### 3. **Comprehensive Training Pipeline**

- Automatic text preprocessing and cleaning
- Train/test data splitting with stratification
- Cross-validation for robust performance estimates
- Detailed performance metrics and evaluation

### 4. **Advanced Features**

- Feature importance analysis for TF-IDF models
- Confidence scores and probability distributions
- Model persistence (save/load functionality)
- Custom category support
- Batch prediction capabilities

## Installation

```bash
# Core dependencies
pip install scikit-learn pandas numpy joblib

# Optional: For transformer models
pip install transformers torch accelerate
```

## Quick Start

### Basic TF-IDF Classification

```python
from email_classifier import EmailClassifier

# Initialize classifier
classifier = EmailClassifier(model_type='tfidf', classifier_type='logistic')

# Training data
emails = [
    "Team meeting scheduled for tomorrow at 2 PM",
    "Happy birthday! Hope you have a great day",
    "Your bank statement is ready for review",
    "50% off sale - limited time only!",
    "You've won a million dollars! Click here",
    "URGENT: Server down, immediate action required"
]

labels = ['Work', 'Personal', 'Finance', 'Promotions', 'Spam', 'Urgent']

# Train the model
metrics = classifier.train(emails, labels)
print(f"Test accuracy: {metrics['test_accuracy']:.3f}")

# Classify new emails
result = classifier.predict("Project deadline moved to next week")
print(f"Category: {result.predicted_category}")
print(f"Confidence: {result.confidence:.3f}")
```

### Transformer-based Classification

```python
# Initialize transformer classifier
transformer_classifier = EmailClassifier(
    model_type='transformer',
    transformer_model='distilbert-base-uncased'
)

# Train (requires more data for optimal performance)
metrics = transformer_classifier.train(
    training_texts,
    training_labels,
    num_epochs=3,
    batch_size=16
)

# Predict with detailed probabilities
result = transformer_classifier.predict(
    "Critical system alert requires immediate attention",
    return_probabilities=True
)

print(f"Predicted: {result.predicted_category}")
print(f"Probabilities: {result.probabilities}")
```

## API Reference

### EmailClassifier Class

#### `__init__(model_type='tfidf', classifier_type='logistic', transformer_model='distilbert-base-uncased', categories=None)`

Initialize the EmailClassifier.

**Parameters:**

- `model_type` (str): 'tfidf' or 'transformer'
- `classifier_type` (str): For TF-IDF: 'logistic' or 'random_forest'
- `transformer_model` (str): HuggingFace model name
- `categories` (List[str]): Custom email categories

#### `train(texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]`

Train the classifier on labeled email data.

**Parameters:**

- `texts`: List of email texts (subjects + bodies recommended)
- `labels`: Corresponding category labels
- `**kwargs`: Model-specific training parameters

**TF-IDF Parameters:**

- `max_features` (int): Maximum TF-IDF features (default: 10000)
- `ngram_range` (tuple): N-gram range (default: (1, 2))
- `test_size` (float): Test set proportion (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Transformer Parameters:**

- `num_epochs` (int): Training epochs (default: 3)
- `batch_size` (int): Batch size (default: 16)
- `learning_rate` (float): Learning rate (default: 2e-5)
- `max_length` (int): Maximum sequence length (default: 512)

**Returns:** Dictionary with training metrics

#### `predict(texts: Union[str, List[str]], return_probabilities=True) -> Union[ClassificationResult, List[ClassificationResult]]`

Predict email categories.

**Parameters:**

- `texts`: Single email text or list of texts
- `return_probabilities`: Include probability scores

**Returns:** ClassificationResult or list of results

#### `evaluate(texts: List[str], true_labels: List[str]) -> Dict[str, Any]`

Evaluate model performance on test data.

#### `save_model(file_path: Union[str, Path])`

Save trained model to disk.

#### `load_model(file_path: Union[str, Path])`

Load trained model from disk.

#### `get_feature_importance(top_n=20) -> Dict[str, List[Tuple[str, float]]]`

Get feature importance for TF-IDF models.

#### `get_model_info() -> Dict[str, Any]`

Get comprehensive model information.

### ClassificationResult Class

Result object returned by predictions:

```python
@dataclass
class ClassificationResult:
    predicted_category: str      # Predicted email category
    confidence: float           # Confidence score (0-1)
    probabilities: Dict[str, float]  # Category probabilities
    processing_time: float      # Prediction time in seconds
```

## Advanced Usage

### Custom Categories

```python
# Define custom categories
custom_categories = ['Important', 'Newsletters', 'Support', 'Social']

classifier = EmailClassifier(
    model_type='tfidf',
    categories=custom_categories
)

# Train with custom labels
classifier.train(custom_texts, custom_labels)
```

### Feature Importance Analysis

```python
# Train TF-IDF model
classifier = EmailClassifier(model_type='tfidf', classifier_type='logistic')
classifier.train(texts, labels)

# Analyze feature importance
importance = classifier.get_feature_importance(top_n=15)

for category, features in importance.items():
    print(f"\nTop features for {category}:")
    for feature, score in features[:10]:
        print(f"  {feature}: {score:.4f}")
```

### Model Comparison

```python
models = [
    ('TF-IDF + Logistic', 'tfidf', 'logistic'),
    ('TF-IDF + Random Forest', 'tfidf', 'random_forest'),
    ('DistilBERT', 'transformer', None)
]

results = []
for name, model_type, classifier_type in models:
    classifier = EmailClassifier(
        model_type=model_type,
        classifier_type=classifier_type
    )
    metrics = classifier.train(texts, labels)
    results.append((name, metrics['test_accuracy']))

# Find best model
best_model = max(results, key=lambda x: x[1])
print(f"Best model: {best_model[0]} (accuracy: {best_model[1]:.3f})")
```

### Batch Processing

```python
# Classify multiple emails efficiently
email_batch = [
    "Project status update needed for client meeting",
    "Birthday party invitation for this weekend",
    "Investment portfolio quarterly report available",
    "Limited time offer - 50% discount on premium plans"
]

# Single batch prediction
results = classifier.predict(email_batch)

for email, result in zip(email_batch, results):
    print(f"'{email[:40]}...' -> {result.predicted_category}")
```

## Text Preprocessing

The EmailClassifier automatically preprocesses text:

1. **Lowercase conversion**
2. **HTML tag removal**
3. **URL replacement** with `[URL]` token
4. **Email address replacement** with `[EMAIL]` token
5. **Whitespace normalization**
6. **Length limiting** (10,000 characters max)

## Model Performance Guidelines

### TF-IDF Models

**Pros:**

- Fast training and prediction
- Low memory requirements
- Good interpretability
- Works well with small datasets

**Cons:**

- Limited context understanding
- May miss semantic relationships
- Performance plateaus with more data

**Recommended for:**

- Real-time classification
- Resource-constrained environments
- Interpretability requirements
- Small to medium datasets (< 10k emails)

### Transformer Models

**Pros:**

- Superior context understanding
- Better semantic comprehension
- State-of-the-art accuracy
- Transfer learning benefits

**Cons:**

- Slower training and inference
- Higher memory requirements
- Less interpretable
- Requires larger datasets

**Recommended for:**

- Maximum accuracy requirements
- Large datasets (> 1k emails)
- Complex email content
- Production systems with adequate resources

## Performance Optimization

### For TF-IDF Models

```python
# Optimize for speed
classifier = EmailClassifier(
    model_type='tfidf',
    classifier_type='logistic'
)

metrics = classifier.train(
    texts, labels,
    max_features=5000,  # Reduce features
    ngram_range=(1, 1)  # Only unigrams
)

# Optimize for accuracy
classifier = EmailClassifier(
    model_type='tfidf',
    classifier_type='random_forest'
)

metrics = classifier.train(
    texts, labels,
    max_features=20000,  # More features
    ngram_range=(1, 3)   # Include trigrams
)
```

### For Transformer Models

```python
# Optimize for speed
classifier = EmailClassifier(
    model_type='transformer',
    transformer_model='distilbert-base-uncased'  # Smaller model
)

metrics = classifier.train(
    texts, labels,
    num_epochs=2,       # Fewer epochs
    batch_size=32,      # Larger batches
    max_length=256      # Shorter sequences
)

# Optimize for accuracy
classifier = EmailClassifier(
    model_type='transformer',
    transformer_model='bert-base-uncased'  # Larger model
)

metrics = classifier.train(
    texts, labels,
    num_epochs=5,       # More epochs
    batch_size=8,       # Smaller batches
    max_length=512      # Longer sequences
)
```

## Error Handling

The EmailClassifier includes comprehensive error handling:

```python
from email_classifier import EmailClassifierError, ModelNotTrainedError

try:
    # Train model
    classifier = EmailClassifier(model_type='tfidf')
    classifier.train(texts, labels)

    # Make predictions
    result = classifier.predict("Test email")

except EmailClassifierError as e:
    print(f"Classification error: {e}")

except ModelNotTrainedError as e:
    print(f"Model not trained: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Integration with EmailDataLoader

```python
from email_loader import EmailDataLoader
from email_classifier import EmailClassifier

# Load training data
loader = EmailDataLoader()
emails = loader.generate_synthetic_emails(count=500)

# Extract text and labels
texts = [f"{email.subject} {email.body}" for email in emails]
labels = [email.category for email in emails]

# Train classifier
classifier = EmailClassifier(model_type='tfidf')
metrics = classifier.train(texts, labels)

# Classify new emails
new_emails = loader.fetch_emails_imap(folder='INBOX', limit=10)
for email in new_emails:
    combined_text = f"{email.subject} {email.body}"
    result = classifier.predict(combined_text)
    print(f"Email: {email.subject} -> {result.predicted_category}")
```

## Best Practices

### 1. **Data Quality**

- Ensure balanced training data across categories
- Include diverse email styles and formats
- Remove or handle duplicate emails
- Validate label consistency

### 2. **Feature Engineering**

- Combine subject and body for better classification
- Consider email metadata (sender, time, etc.)
- Handle multilingual content appropriately
- Preprocess consistently between training and prediction

### 3. **Model Selection**

- Start with TF-IDF for baseline performance
- Use transformers for complex classification tasks
- Consider computational constraints
- Evaluate multiple models systematically

### 4. **Evaluation**

- Use cross-validation for robust estimates
- Test on held-out data from different time periods
- Monitor performance on edge cases
- Track performance over time in production

### 5. **Production Deployment**

- Implement model versioning
- Monitor prediction confidence scores
- Set up retraining pipelines
- Handle unknown categories gracefully

## Troubleshooting

### Common Issues

**Low Accuracy:**

- Increase training data size
- Check label quality and consistency
- Try different model types
- Adjust hyperparameters

**Slow Performance:**

- Reduce TF-IDF features
- Use smaller transformer models
- Implement batch processing
- Optimize hardware resources

**Memory Issues:**

- Reduce batch size for transformers
- Limit sequence length
- Use gradient checkpointing
- Consider model distillation

**Import Errors:**

- Install required dependencies
- Check Python version compatibility
- Verify CUDA setup for GPU usage
- Update package versions

### Getting Help

For issues and feature requests:

1. Check the comprehensive test suite in `test_email_classifier.py`
2. Review example usage in `email_classifier_example.py`
3. Examine integration patterns in `integration_example.py`
4. Consult the MailMate documentation

The EmailClassifier is designed to be robust, scalable, and easy to integrate into the broader MailMate ecosystem while providing state-of-the-art email classification capabilities.
