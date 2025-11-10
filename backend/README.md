# MailMate Backend - AI Email Management System

## Overview

The MailMate backend provides a comprehensive suite of Python classes for AI-driven email management, including data loading, classification, summarization, and text-to-speech capabilities. This system is designed for sophisticated email processing workflows with proper error handling and extensibility.

## Core Components

### 1. **EmailDataLoader** - Email Data Management
- Load emails from CSV files using pandas
- Fetch emails via IMAP from Gmail, Outlook, and other providers
- Generate synthetic email samples for testing and development
- Export to CSV/JSON formats with proper serialization

### 2. **EmailClassifier** - Machine Learning Classification
- TF-IDF vectorization with scikit-learn
- Multiple classification algorithms (Logistic Regression, Random Forest)
- Optional transformer-based models using HuggingFace BERT/DistilBERT
- Confidence scoring and probability distributions
- Model persistence and loading capabilities

### 3. **EmailSummarizer** - AI-Powered Summarization
- HuggingFace transformer models (BART, Pegasus, T5)
- Configurable summary lengths and quality settings
- Batch processing for multiple emails
- Graceful fallback when transformers unavailable

### 4. **TextToSpeech** - Audio Conversion
- Multiple TTS engines (gTTS, pyttsx3) with automatic selection
- Audio format support (MP3, WAV)
- Voice customization and language support
- Email summary audio integration
- Batch processing and audio playback capabilities

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: For enhanced ML capabilities
pip install transformers torch

# Optional: For audio functionality
pip install gtts pyttsx3 pygame
```

## Quick Start

```python
from email_loader import EmailDataLoader
from email_classifier import EmailClassifier
from email_summarizer import EmailSummarizer
from text_to_speech import TextToSpeech

# Complete email processing pipeline
loader = EmailDataLoader()
classifier = EmailClassifier()
summarizer = EmailSummarizer()
tts = TextToSpeech()

# Generate sample emails
emails = loader.generate_synthetic_emails(count=100)

# Train classifier
classifier.train(emails)

# Process new email
classification = classifier.predict(email_content)
summary = summarizer.summarize_email(email_content)

# Convert to audio
audio_result = tts.convert_text_to_speech(
    summary.summary, 
    "audio_outputs/summary.mp3"
)
```

## Features

### 1. **CSV File Loading**

- Load emails from CSV files using pandas
- Support for custom column mappings
- Comprehensive data validation
- Automatic data type conversion
- Error handling for malformed CSV files

### 2. **IMAP Email Fetching**

- Connect to Gmail, Outlook, Yahoo, and iCloud via IMAP
- OAuth2 support (framework ready)
- Folder management and listing
- Email parsing with attachment extraction
- Proper SSL/TLS security
- Connection management with context managers

### 3. **Synthetic Email Generation**

- Generate realistic test emails for development
- Customizable categories (Personal, Work, Finance, etc.)
- Configurable priority levels
- Realistic email addresses and content
- Attachment simulation
- Timestamp generation within date ranges

### 4. **Data Export**

- Export to CSV format with pandas
- Export to JSON format with proper serialization
- Preserve data integrity across export/import cycles
- Configurable encoding and formatting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from email_loader import EmailDataLoader

# Initialize loader
loader = EmailDataLoader()

# Generate synthetic emails for testing
emails = loader.generate_synthetic_emails(count=50)

# Export to CSV
loader.export_to_csv(emails, "sample_emails.csv")

# Load from CSV
loaded_emails = loader.load_from_csv("sample_emails.csv")

# IMAP example (requires credentials)
loader.connect_imap("your_email@gmail.com", "app_password", "gmail")
imap_emails = loader.fetch_emails_imap(folder='INBOX', limit=10)
loader.disconnect_imap()
```

## Classes

### EmailData

A dataclass representing an email message with the following fields:

- `id: str` - Unique email identifier
- `from_address: str` - Sender email address
- `to_address: str` - Recipient email address
- `cc_address: Optional[str]` - CC recipients
- `bcc_address: Optional[str]` - BCC recipients
- `subject: str` - Email subject line
- `body: str` - Email body content
- `timestamp: datetime` - Email timestamp
- `category: Optional[str]` - Email category (Personal, Work, etc.)
- `priority: Optional[str]` - Priority level (Low, Medium, High, etc.)
- `attachments: List[str]` - List of attachment filenames
- `is_read: bool` - Read status
- `folder: str` - Source folder name

### EmailDataLoader

Main class for loading emails from various sources.

#### Methods

##### `__init__(config: Optional[Dict[str, Any]] = None)`

Initialize the EmailDataLoader with optional configuration.

##### `load_from_csv(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[EmailData]`

Load emails from a CSV file.

**Parameters:**

- `file_path`: Path to CSV file
- `encoding`: File encoding (default: utf-8)

**Returns:** List of EmailData objects

**Raises:**

- `CSVFormatError`: Invalid CSV format
- `FileNotFoundError`: File doesn't exist

##### `connect_imap(email_address: str, password: str, provider: str = 'gmail', custom_server: Optional[tuple] = None)`

Connect to IMAP server.

**Parameters:**

- `email_address`: Email address for authentication
- `password`: Password or app-specific password
- `provider`: Email provider ('gmail', 'outlook', 'yahoo', 'icloud')
- `custom_server`: Optional (server, port) tuple for custom servers

**Raises:** `IMAPConnectionError`: Connection fails

##### `fetch_emails_imap(folder: str = 'INBOX', limit: int = 100, search_criteria: str = 'ALL') -> List[EmailData]`

Fetch emails from IMAP server.

**Parameters:**

- `folder`: IMAP folder name
- `limit`: Maximum emails to fetch
- `search_criteria`: IMAP search criteria

**Returns:** List of EmailData objects

##### `generate_synthetic_emails(count: int = 100, categories: Optional[List[str]] = None, priorities: Optional[List[str]] = None) -> List[EmailData]`

Generate synthetic email samples for testing.

**Parameters:**

- `count`: Number of emails to generate
- `categories`: List of categories to use
- `priorities`: List of priorities to use

**Returns:** List of synthetic EmailData objects

##### `export_to_csv(emails: List[EmailData], file_path: Union[str, Path], encoding: str = 'utf-8')`

Export emails to CSV file.

##### `export_to_json(emails: List[EmailData], file_path: Union[str, Path], encoding: str = 'utf-8', indent: int = 2)`

Export emails to JSON file.

##### `get_folder_list() -> List[str]`

Get list of available IMAP folders.

##### `disconnect_imap()`

Disconnect from IMAP server.

## Configuration

The EmailDataLoader accepts a configuration dictionary to customize behavior:

```python
config = {
    'csv_column_mapping': {
        'id': 'email_id',
        'from_address': 'sender',
        'to_address': 'recipient',
        'subject': 'email_subject',
        'body': 'content',
        'timestamp': 'date_time',
        'category': 'type',
        'priority': 'importance'
    }
}

loader = EmailDataLoader(config=config)
```

## Error Handling

The module defines custom exceptions for different error scenarios:

- `EmailDataLoaderError`: Base exception for all loader errors
- `IMAPConnectionError`: IMAP connection and authentication errors
- `CSVFormatError`: CSV format and parsing errors

## IMAP Security

For Gmail, use App Passwords instead of your regular password:

1. Enable 2-Factor Authentication
2. Generate an App Password in Google Account settings
3. Use the App Password with the EmailDataLoader

## Context Manager Support

The EmailDataLoader supports context manager usage for automatic cleanup:

```python
with EmailDataLoader() as loader:
    loader.connect_imap("email@gmail.com", "password", "gmail")
    emails = loader.fetch_emails_imap()
    # Connection automatically closed when exiting context
```

## Testing

Run the comprehensive test suite:

```bash
cd tests
python test_email_loader.py
```

The test suite includes:

- Unit tests for all methods
- Integration tests for complete workflows
- Error handling verification
- Mock IMAP server testing
- Data integrity validation

## Future Enhancements

- OAuth2 integration for Gmail and Outlook
- Batch processing for large email sets
- Email content preprocessing for ML pipelines
- Support for additional email providers
- Email threading and conversation grouping
- Advanced search and filtering capabilities
