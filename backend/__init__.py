"""
MailMate Backend Package

This package contains the core backend modules for the MailMate AI-driven email management system.

Modules:
- email_loader: EmailDataLoader class for loading emails from multiple sources
- email_classifier: EmailClassifier class for ML-based email classification
- email_summarizer: EmailSummarizer class for AI-powered email summarization
- tts: Text-to-speech functionality (to be implemented)
"""

from .email_loader import EmailDataLoader, EmailData, EmailDataLoaderError, IMAPConnectionError, CSVFormatError
from .email_classifier import EmailClassifier, ClassificationResult, EmailClassifierError, ModelNotTrainedError, EMAIL_CATEGORIES
from .email_summarizer import EmailSummarizer, SummaryResult, EmailSummarizerError, ModelNotAvailableError, TextTooShortError

__version__ = "0.1.0"
__author__ = "MailMate Development Team"

__all__ = [
    "EmailDataLoader",
    "EmailData", 
    "EmailDataLoaderError",
    "IMAPConnectionError",
    "CSVFormatError",
    "EmailClassifier",
    "ClassificationResult",
    "EmailClassifierError",
    "ModelNotTrainedError",
    "EMAIL_CATEGORIES",
    "EmailSummarizer",
    "SummaryResult",
    "EmailSummarizerError",
    "ModelNotAvailableError",
    "TextTooShortError"
]