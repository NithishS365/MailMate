"""
EmailDataLoader module for MailMate - Sophisticated AI-driven email management system.

This module provides the EmailDataLoader class that supports:
- Loading email data from CSV files using pandas
- Fetching emails via IMAP from Gmail or Outlook with OAuth2 support
- Generating synthetic email samples for testing
- Comprehensive error handling and type hints
"""

import csv
import email.message
import imaplib
import json
import logging
import random
import ssl
from datetime import datetime, timedelta
from email.header import decode_header
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import os

import pandas as pd
from faker import Faker
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Configure logging
logger = logging.getLogger(__name__)

# Email server configurations
IMAP_SERVERS = {
    'gmail': ('imap.gmail.com', 993),
    'outlook': ('outlook.office365.com', 993),
    'yahoo': ('imap.mail.yahoo.com', 993),
    'icloud': ('imap.mail.me.com', 993),
}

@dataclass
class EmailData:
    """Data class representing an email message."""
    id: str
    from_address: str
    to_address: str
    cc_address: Optional[str] = None
    bcc_address: Optional[str] = None
    subject: str = ""
    body: str = ""
    timestamp: datetime = None
    category: Optional[str] = None
    priority: Optional[str] = None
    attachments: List[str] = None
    is_read: bool = False
    folder: str = "INBOX"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.attachments is None:
            self.attachments = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EmailData to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailData':
        """Create EmailData from dictionary."""
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EmailDataLoaderError(Exception):
    """Base exception for EmailDataLoader errors."""
    pass


class IMAPConnectionError(EmailDataLoaderError):
    """Exception raised when IMAP connection fails."""
    pass


class CSVFormatError(EmailDataLoaderError):
    """Exception raised when CSV format is invalid."""
    pass


class EmailDataLoader:
    """
    EmailDataLoader class for loading emails from multiple sources.
    
    Supports:
    - CSV file loading with pandas
    - IMAP email fetching from Gmail, Outlook, etc.
    - Synthetic email generation for testing
    - Loading scraped email data from JSON files
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EmailDataLoader.
        
        Args:
            config: Optional configuration dictionary containing IMAP settings,
                   CSV column mappings, etc.
        """
        self.config = config or {}
        self.fake = Faker()
        self._imap_connection: Optional[imaplib.IMAP4_SSL] = None
        
        # Default CSV column mappings
        self.csv_column_mapping = self.config.get('csv_column_mapping', {
            'id': 'id',
            'from_address': 'from',
            'to_address': 'to',
            'cc_address': 'cc',
            'bcc_address': 'bcc',
            'subject': 'subject',
            'body': 'body',
            'timestamp': 'timestamp',
            'category': 'category',
            'priority': 'priority',
            'is_read': 'is_read',
            'folder': 'folder'
        })

    def load_scraped_emails(self, file_path: Union[str, Path]) -> List[EmailData]:
        """
        Load scraped emails from JSON file.
        
        Args:
            file_path: Path to the scraped emails JSON file
            
        Returns:
            List of EmailData objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Scraped emails file not found: {file_path}")
        
        logger.info(f"Loading scraped emails from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                email_dicts = json.load(f)
            
            if not isinstance(email_dicts, list):
                raise ValueError("Invalid file format: expected list of email objects")
            
            emails = []
            for email_dict in email_dicts:
                try:
                    # Convert timestamp string back to datetime
                    if 'timestamp' in email_dict and email_dict['timestamp']:
                        email_dict['timestamp'] = datetime.fromisoformat(email_dict['timestamp'].replace('Z', '+00:00'))
                    
                    # Create EmailData object
                    email_obj = EmailData(**email_dict)
                    emails.append(email_obj)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse email object: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(emails)} scraped emails")
            return emails
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise EmailDataLoaderError(f"Failed to load scraped emails: {e}")

    def get_latest_scraped_file(self, data_dir: Union[str, Path] = 'data') -> Optional[Path]:
        """
        Get the most recent scraped emails file.
        
        Args:
            data_dir: Directory containing scraped email files
            
        Returns:
            Path to latest file or None if no files found
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return None
        
        # Look for files matching the pattern
        pattern = 'gmail_emails_*.json'
        files = list(data_dir.glob(pattern))
        
        if not files:
            # Also check for manual files
            json_files = list(data_dir.glob('*.json'))
            if json_files:
                files = json_files
        
        if not files:
            return None
        
        # Return the most recent file
        return max(files, key=lambda f: f.stat().st_mtime)

    def load_emails_smart(self, 
                         prefer_scraped: bool = True, 
                         data_dir: Union[str, Path] = 'data',
                         fallback_count: int = 50) -> List[EmailData]:
        """
        Smart email loading that tries scraped data first, then falls back to synthetic.
        
        Args:
            prefer_scraped: Whether to prefer scraped data over synthetic
            data_dir: Directory to look for scraped email files
            fallback_count: Number of synthetic emails to generate if no scraped data
            
        Returns:
            List of EmailData objects
        """
        if prefer_scraped:
            # Try to load scraped emails first
            latest_file = self.get_latest_scraped_file(data_dir)
            if latest_file:
                try:
                    emails = self.load_scraped_emails(latest_file)
                    if emails:
                        logger.info(f"Loaded {len(emails)} scraped emails from {latest_file}")
                        return emails
                except Exception as e:
                    logger.warning(f"Failed to load scraped emails: {e}")
        
        # Fallback to synthetic emails
        logger.info(f"Generating {fallback_count} synthetic emails as fallback")
        return self.generate_synthetic_emails(fallback_count)
    
    def load_from_csv(self, file_path: Union[str, Path], 
                      encoding: str = 'utf-8') -> List[EmailData]:
        """
        Load email data from a CSV file using pandas.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of EmailData objects
            
        Raises:
            CSVFormatError: If CSV format is invalid
            FileNotFoundError: If file doesn't exist
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            logger.info(f"Loading emails from CSV: {file_path}")
            
            # Read CSV with pandas
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Validate required columns
            required_columns = ['from_address', 'to_address', 'subject']
            mapped_columns = [self.csv_column_mapping.get(col, col) for col in required_columns]
            missing_columns = [col for col in mapped_columns if col not in df.columns]
            
            if missing_columns:
                raise CSVFormatError(
                    f"Missing required columns in CSV: {missing_columns}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            emails = []
            for index, row in df.iterrows():
                try:
                    # Map CSV columns to EmailData fields
                    email_data = {}
                    for field, csv_col in self.csv_column_mapping.items():
                        if csv_col in df.columns and pd.notna(row[csv_col]):
                            email_data[field] = row[csv_col]
                    
                    # Generate ID if not present
                    if 'id' not in email_data:
                        email_data['id'] = f"csv_{index}_{hash(str(row.to_dict()))}"
                    
                    # Parse timestamp if present
                    if 'timestamp' in email_data:
                        try:
                            email_data['timestamp'] = pd.to_datetime(email_data['timestamp'])
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid timestamp in row {index}: {email_data['timestamp']}")
                            email_data['timestamp'] = datetime.now()
                    
                    # Handle attachments (assume comma-separated string)
                    if 'attachments' in email_data and isinstance(email_data['attachments'], str):
                        email_data['attachments'] = [att.strip() for att in email_data['attachments'].split(',') if att.strip()]
                    
                    emails.append(EmailData(**email_data))
                    
                except Exception as e:
                    logger.warning(f"Failed to parse email at row {index}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(emails)} emails from CSV")
            return emails
            
        except pd.errors.EmptyDataError:
            raise CSVFormatError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise CSVFormatError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise EmailDataLoaderError(f"Failed to load CSV: {e}")
    
    def authenticate_oauth2(self, client_secret_file: str = 'config/client_secret.json',
                          token_file: str = 'config/token.pickle',
                          scopes: List[str] = None) -> Credentials:
        """
        Authenticate using OAuth2 for Gmail.
        
        Args:
            client_secret_file: Path to the OAuth2 client secret file
            token_file: Path to store/retrieve the token
            scopes: List of required scopes. Defaults to Gmail read-only
            
        Returns:
            Google OAuth2 credentials
            
        Raises:
            IMAPConnectionError: If authentication fails
        """
        if scopes is None:
            scopes = ['https://www.googleapis.com/auth/gmail.readonly']
            
        try:
            creds = None
            # Load existing token
            if os.path.exists(token_file):
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)

            # Refresh token if expired
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            
            # Get new token if needed
            if not creds or not creds.valid:
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
                creds = flow.run_local_server(port=8080)
                
                # Save token
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            return creds
            
        except Exception as e:
            raise IMAPConnectionError(f"OAuth2 authentication failed: {e}")

    def connect_imap(self, email_address: str, password: str = None, 
                     provider: str = 'gmail', custom_server: Optional[tuple] = None,
                     use_oauth2: bool = False, client_secret_file: str = None,
                     token_file: str = None) -> None:
        """
        Connect to IMAP server.
        
        Args:
            email_address: Email address for authentication
            password: Password or app-specific password (not needed for OAuth2)
            provider: Email provider ('gmail', 'outlook', 'yahoo', 'icloud')
            custom_server: Optional tuple of (server, port) for custom IMAP servers
            use_oauth2: Whether to use OAuth2 authentication
            client_secret_file: Path to OAuth2 client secret file
            token_file: Path to store/retrieve OAuth2 token
            
        Raises:
            IMAPConnectionError: If connection fails
        """
        try:
            if custom_server:
                server, port = custom_server
            else:
                if provider not in IMAP_SERVERS:
                    raise IMAPConnectionError(f"Unsupported provider: {provider}")
                server, port = IMAP_SERVERS[provider]
            
            logger.info(f"Connecting to IMAP server: {server}:{port}")
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect to server
            self._imap_connection = imaplib.IMAP4_SSL(server, port, ssl_context=context)
            
            if use_oauth2 and provider == 'gmail':
                # Use OAuth2 for Gmail
                if not client_secret_file:
                    client_secret_file = 'config/client_secret.json'
                if not token_file:
                    token_file = 'config/token.pickle'
                
                creds = self.authenticate_oauth2(
                    client_secret_file=client_secret_file,
                    token_file=token_file,
                    scopes=['https://www.googleapis.com/auth/gmail.readonly',
                           'https://www.googleapis.com/auth/gmail.modify']
                )
                
                auth_string = f'user={email_address}\1auth=Bearer {creds.token}\1\1'
                self._imap_connection.authenticate('XOAUTH2', lambda x: auth_string)
                logger.info(f"Successfully connected to Gmail using OAuth2 for {email_address}")
            else:
                # Traditional password login
                if not password:
                    raise IMAPConnectionError("Password required for non-OAuth2 connection")
                self._imap_connection.login(email_address, password)
                logger.info(f"Successfully connected to {provider} IMAP for {email_address}")
            
        except imaplib.IMAP4.error as e:
            raise IMAPConnectionError(f"IMAP authentication failed: {e}")
        except Exception as e:
            raise IMAPConnectionError(f"Failed to connect to IMAP server: {e}")
    
    def disconnect_imap(self) -> None:
        """Disconnect from IMAP server."""
        if self._imap_connection:
            try:
                self._imap_connection.close()
                self._imap_connection.logout()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.warning(f"Error during IMAP disconnection: {e}")
            finally:
                self._imap_connection = None
    
    def fetch_emails_imap(self, folder: str = 'INBOX', limit: int = 100,
                          search_criteria: str = 'ALL') -> List[EmailData]:
        """
        Fetch emails from IMAP server.
        
        Args:
            folder: IMAP folder to fetch from (default: INBOX)
            limit: Maximum number of emails to fetch
            search_criteria: IMAP search criteria (default: ALL)
            
        Returns:
            List of EmailData objects
            
        Raises:
            IMAPConnectionError: If not connected to IMAP server
        """
        if not self._imap_connection:
            raise IMAPConnectionError("Not connected to IMAP server. Call connect_imap() first.")
        
        try:
            logger.info(f"Fetching emails from folder: {folder}")
            
            # Select folder
            status, messages = self._imap_connection.select(folder)
            if status != 'OK':
                raise IMAPConnectionError(f"Failed to select folder {folder}: {messages}")
            
            # Search for emails
            status, email_ids = self._imap_connection.search(None, search_criteria)
            if status != 'OK':
                raise IMAPConnectionError(f"Failed to search emails: {email_ids}")
            
            email_id_list = email_ids[0].split()
            
            # Limit the number of emails
            if limit > 0:
                email_id_list = email_id_list[-limit:]  # Get most recent emails
            
            emails = []
            for email_id in email_id_list:
                try:
                    # Fetch email
                    status, email_data = self._imap_connection.fetch(email_id, '(RFC822)')
                    if status != 'OK':
                        logger.warning(f"Failed to fetch email {email_id}")
                        continue
                    
                    # Parse email
                    raw_email = email_data[0][1]
                    email_message = email.message_from_bytes(raw_email)
                    
                    # Extract email data
                    parsed_email = self._parse_email_message(email_message, email_id.decode(), folder)
                    if parsed_email:
                        emails.append(parsed_email)
                        
                except Exception as e:
                    logger.warning(f"Failed to process email {email_id}: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(emails)} emails from {folder}")
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            raise IMAPConnectionError(f"Failed to fetch emails: {e}")
    
    def _parse_email_message(self, email_message: email.message.EmailMessage, 
                           email_id: str, folder: str) -> Optional[EmailData]:
        """
        Parse an email message into EmailData object.
        
        Args:
            email_message: Email message object
            email_id: Email ID
            folder: Folder name
            
        Returns:
            EmailData object or None if parsing fails
        """
        try:
            # Decode header
            def decode_mime_words(s):
                return ''.join(
                    word.decode(encoding or 'utf-8') if isinstance(word, bytes) else word
                    for word, encoding in decode_header(s)
                )
            
            # Extract basic fields
            from_address = decode_mime_words(email_message.get('From', ''))
            to_address = decode_mime_words(email_message.get('To', ''))
            cc_address = decode_mime_words(email_message.get('Cc', '')) if email_message.get('Cc') else None
            bcc_address = decode_mime_words(email_message.get('Bcc', '')) if email_message.get('Bcc') else None
            subject = decode_mime_words(email_message.get('Subject', ''))
            
            # Parse timestamp
            timestamp_str = email_message.get('Date')
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = email.utils.parsedate_to_datetime(timestamp_str)
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Extract body
            body = self._extract_email_body(email_message)
            
            # Extract attachments
            attachments = self._extract_attachments(email_message)
            
            return EmailData(
                id=f"imap_{email_id}",
                from_address=from_address,
                to_address=to_address,
                cc_address=cc_address,
                bcc_address=bcc_address,
                subject=subject,
                body=body,
                timestamp=timestamp,
                attachments=attachments,
                folder=folder
            )
            
        except Exception as e:
            logger.error(f"Failed to parse email message: {e}")
            return None
    
    def _extract_email_body(self, email_message: email.message.EmailMessage) -> str:
        """Extract email body from email message."""
        body = ""
        
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    if content_type == "text/plain":
                        charset = part.get_content_charset() or 'utf-8'
                        part_body = part.get_payload(decode=True).decode(charset, errors='ignore')
                        body += part_body
                    elif content_type == "text/html" and not body:
                        # Use HTML as fallback if no plain text
                        charset = part.get_content_charset() or 'utf-8'
                        part_body = part.get_payload(decode=True).decode(charset, errors='ignore')
                        body = part_body
            else:
                charset = email_message.get_content_charset() or 'utf-8'
                body = email_message.get_payload(decode=True).decode(charset, errors='ignore')
                
        except Exception as e:
            logger.warning(f"Failed to extract email body: {e}")
            body = "Failed to extract email body"
        
        return body.strip()
    
    def _extract_attachments(self, email_message: email.message.EmailMessage) -> List[str]:
        """Extract attachment names from email message."""
        attachments = []
        
        try:
            for part in email_message.walk():
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        # Decode filename if needed
                        filename = ''.join(
                            word.decode(encoding or 'utf-8') if isinstance(word, bytes) else word
                            for word, encoding in decode_header(filename)
                        )
                        attachments.append(filename)
        except Exception as e:
            logger.warning(f"Failed to extract attachments: {e}")
        
        return attachments
    
    def generate_synthetic_emails(self, count: int = 100, 
                                categories: Optional[List[str]] = None,
                                priorities: Optional[List[str]] = None) -> List[EmailData]:
        """
        Generate synthetic email samples for testing.
        
        Args:
            count: Number of emails to generate
            categories: List of categories to assign (default: common categories)
            priorities: List of priorities to assign (default: common priorities)
            
        Returns:
            List of synthetic EmailData objects
        """
        if categories is None:
            categories = ['Personal', 'Work', 'Finance', 'Promotions', 'Spam', 'Urgent']
        
        if priorities is None:
            priorities = ['Very Low', 'Low', 'Medium', 'High', 'Critical']
        
        # Email templates for different categories
        templates = {
            'Personal': {
                'subjects': [
                    "Happy Birthday!",
                    "Weekend plans?",
                    "Family reunion next month",
                    "How are you doing?",
                    "Quick catch up"
                ],
                'bodies': [
                    "Hey! Hope you're doing well. Just wanted to catch up and see how things are going.",
                    "I was thinking about our conversation the other day. What do you think about meeting up this weekend?",
                    "Family is planning a reunion next month. Would love to have you there!",
                    "Just checking in to see how you're doing. It's been a while since we last talked."
                ]
            },
            'Work': {
                'subjects': [
                    "Project update - Q4 deliverables",
                    "Meeting scheduled for tomorrow",
                    "Budget approval needed",
                    "Team performance review",
                    "Client presentation feedback"
                ],
                'bodies': [
                    "Please find attached the latest project update. We need to discuss the Q4 deliverables in our next meeting.",
                    "I've scheduled a meeting for tomorrow at 2 PM to discuss the upcoming project milestones.",
                    "The budget proposal requires your approval before we can proceed with the implementation.",
                    "Time for our quarterly team performance review. Please prepare your self-assessment."
                ]
            },
            'Finance': {
                'subjects': [
                    "Monthly bank statement",
                    "Investment portfolio update",
                    "Credit card payment due",
                    "Tax document ready",
                    "Insurance premium notice"
                ],
                'bodies': [
                    "Your monthly bank statement is now available for download from your online banking portal.",
                    "Your investment portfolio has performed well this quarter. Review the attached summary.",
                    "This is a reminder that your credit card payment is due in 3 days.",
                    "Your tax documents are ready for download. Please log in to your account to access them."
                ]
            },
            'Promotions': {
                'subjects': [
                    "50% off everything - Limited time!",
                    "Exclusive deals just for you",
                    "Flash sale - 24 hours only",
                    "New arrivals - Check them out",
                    "Member appreciation sale"
                ],
                'bodies': [
                    "Don't miss out on our biggest sale of the year! 50% off everything for a limited time only.",
                    "As a valued customer, you get exclusive access to these amazing deals before anyone else.",
                    "Flash sale alert! Everything must go in the next 24 hours. Shop now before it's too late.",
                    "Check out our latest arrivals and be the first to get your hands on these amazing products."
                ]
            },
            'Spam': {
                'subjects': [
                    "You've won $1,000,000!!!",
                    "Claim your prize now",
                    "Urgent: Verify your account",
                    "Amazing weight loss secret",
                    "Work from home opportunity"
                ],
                'bodies': [
                    "Congratulations! You've won our grand prize of $1,000,000. Click here to claim your prize now!",
                    "Your account needs immediate verification or it will be suspended. Click here to verify now.",
                    "Doctors hate this one simple trick that helps you lose weight fast without diet or exercise.",
                    "Make $5000 a week working from home. No experience necessary. Apply now!"
                ]
            },
            'Urgent': {
                'subjects': [
                    "URGENT: Server down",
                    "Emergency meeting called",
                    "Critical security alert",
                    "Immediate action required",
                    "System maintenance - NOW"
                ],
                'bodies': [
                    "URGENT: Our main server is down and affecting customer operations. Need immediate attention.",
                    "Emergency meeting has been called for 3 PM today. All team leads must attend.",
                    "Critical security vulnerability detected. Immediate patching required on all systems.",
                    "Immediate action is required to resolve the ongoing production issue."
                ]
            }
        }
        
        logger.info(f"Generating {count} synthetic emails")
        
        emails = []
        for i in range(count):
            # Random category and priority
            category = random.choice(categories)
            priority = random.choice(priorities)
            
            # Get templates for category
            category_templates = templates.get(category, templates['Personal'])
            
            # Generate email data
            subject = random.choice(category_templates['subjects'])
            body = random.choice(category_templates['bodies'])
            
            # Add some variation to subject and body
            if random.random() < 0.3:  # 30% chance to add variation
                subject += f" - {self.fake.word().title()}"
            
            # Generate timestamp (within last 30 days)
            timestamp = self.fake.date_time_between(start_date='-30d', end_date='now')
            
            # Generate email addresses
            from_domain = random.choice(['gmail.com', 'outlook.com', 'company.com', 'example.org'])
            to_domain = random.choice(['gmail.com', 'outlook.com', 'mycompany.com'])
            
            from_address = f"{self.fake.user_name()}@{from_domain}"
            to_address = f"{self.fake.user_name()}@{to_domain}"
            
            # Sometimes add CC
            cc_address = None
            if random.random() < 0.2:  # 20% chance of CC
                cc_address = f"{self.fake.user_name()}@{to_domain}"
            
            # Sometimes add attachments
            attachments = []
            if random.random() < 0.15:  # 15% chance of attachments
                attachment_types = ['.pdf', '.docx', '.xlsx', '.png', '.jpg']
                num_attachments = random.randint(1, 3)
                for _ in range(num_attachments):
                    filename = f"{self.fake.word()}{random.choice(attachment_types)}"
                    attachments.append(filename)
            
            email_data = EmailData(
                id=f"synthetic_{i+1}",
                from_address=from_address,
                to_address=to_address,
                cc_address=cc_address,
                subject=subject,
                body=body,
                timestamp=timestamp,
                category=category,
                priority=priority,
                attachments=attachments,
                is_read=random.choice([True, False]),
                folder='INBOX'
            )
            
            emails.append(email_data)
        
        logger.info(f"Successfully generated {len(emails)} synthetic emails")
        return emails
    
    def export_to_csv(self, emails: List[EmailData], file_path: Union[str, Path],
                      encoding: str = 'utf-8') -> None:
        """
        Export emails to CSV file.
        
        Args:
            emails: List of EmailData objects to export
            file_path: Path to output CSV file
            encoding: File encoding (default: utf-8)
        """
        try:
            file_path = Path(file_path)
            
            logger.info(f"Exporting {len(emails)} emails to CSV: {file_path}")
            
            # Convert emails to dictionaries
            email_dicts = []
            for email_data in emails:
                email_dict = email_data.to_dict()
                # Convert list to comma-separated string for CSV
                if email_dict['attachments']:
                    email_dict['attachments'] = ', '.join(email_dict['attachments'])
                else:
                    email_dict['attachments'] = ''
                email_dicts.append(email_dict)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(email_dicts)
            df.to_csv(file_path, index=False, encoding=encoding)
            
            logger.info(f"Successfully exported emails to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export emails to CSV: {e}")
            raise EmailDataLoaderError(f"Failed to export to CSV: {e}")
    
    def export_to_json(self, emails: List[EmailData], file_path: Union[str, Path],
                       encoding: str = 'utf-8', indent: int = 2) -> None:
        """
        Export emails to JSON file.
        
        Args:
            emails: List of EmailData objects to export
            file_path: Path to output JSON file
            encoding: File encoding (default: utf-8)
            indent: JSON indentation (default: 2)
        """
        try:
            file_path = Path(file_path)
            
            logger.info(f"Exporting {len(emails)} emails to JSON: {file_path}")
            
            # Convert emails to dictionaries
            email_dicts = [email_data.to_dict() for email_data in emails]
            
            # Save to JSON
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(email_dicts, f, indent=indent, default=str)
            
            logger.info(f"Successfully exported emails to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export emails to JSON: {e}")
            raise EmailDataLoaderError(f"Failed to export to JSON: {e}")
    
    def get_folder_list(self) -> List[str]:
        """
        Get list of available IMAP folders.
        
        Returns:
            List of folder names
            
        Raises:
            IMAPConnectionError: If not connected to IMAP server
        """
        if not self._imap_connection:
            raise IMAPConnectionError("Not connected to IMAP server. Call connect_imap() first.")
        
        try:
            status, folders = self._imap_connection.list()
            if status != 'OK':
                raise IMAPConnectionError(f"Failed to get folder list: {folders}")
            
            folder_names = []
            for folder in folders:
                # Parse folder name from IMAP response
                # Format: (flags) "delimiter" "folder_name"
                parts = folder.decode().split('"')
                if len(parts) >= 3:
                    folder_name = parts[-2]
                    folder_names.append(folder_name)
            
            return folder_names
            
        except Exception as e:
            logger.error(f"Error getting folder list: {e}")
            raise IMAPConnectionError(f"Failed to get folder list: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from IMAP if connected."""
        self.disconnect_imap()