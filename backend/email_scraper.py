"""
Gmail Email Scraper for MailMate

This module provides comprehensive email scraping functionality from Gmail using OAuth2.
It can fetch all emails from specified folders and store them in various formats.
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import asdict
import imaplib
import email
from email.header import decode_header
import re

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from email_loader import EmailData, EmailDataLoader, IMAPConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailScraper:
    """
    Comprehensive Gmail email scraper with OAuth2 authentication.
    
    Features:
    - Full mailbox scraping
    - Incremental sync
    - Multiple folder support
    - Data persistence
    - Progress tracking
    - Error recovery
    """
    
    def __init__(self, 
                 client_secret_file: str = 'config/client_secret.json',
                 token_file: str = 'config/gmail_token.pickle',
                 data_dir: str = 'data'):
        """
        Initialize the email scraper.
        
        Args:
            client_secret_file: Path to OAuth2 client secret file
            token_file: Path to store OAuth2 tokens
            data_dir: Directory to store scraped email data
        """
        self.client_secret_file = client_secret_file
        self.token_file = token_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Scraping state
        self.email_address = None
        self.imap_connection = None
        self.scraped_emails = []
        self.sync_state_file = self.data_dir / 'sync_state.json'
        self.sync_state = self._load_sync_state()
        
        # Gmail OAuth2 scopes
        self.scopes = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.modify'
        ]
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_new': 0,
            'total_folders': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }

    def _load_sync_state(self) -> Dict:
        """Load synchronization state from file."""
        if self.sync_state_file.exists():
            try:
                with open(self.sync_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
        
        return {
            'last_sync': None,
            'folder_uids': {},  # folder_name -> last_uid
            'total_emails': 0,
            'email_index': {}  # message_id -> file_path
        }

    def _save_sync_state(self):
        """Save synchronization state to file."""
        try:
            with open(self.sync_state_file, 'w') as f:
                json.dump(self.sync_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")

    def authenticate(self) -> Credentials:
        """
        Authenticate with Gmail using OAuth2.
        
        Returns:
            Google OAuth2 credentials
        """
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)

        # Refresh or get new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.client_secret_file):
                    raise IMAPConnectionError(f"Client secret file not found: {self.client_secret_file}")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_file, self.scopes)
                creds = flow.run_local_server(port=8080)
                
                # Save token
                with open(self.token_file, 'wb') as token:
                    pickle.dump(creds, token)
        
        return creds

    def connect_gmail(self, email_address: str):
        """
        Connect to Gmail IMAP server.
        
        Args:
            email_address: Gmail address to connect to
        """
        self.email_address = email_address
        
        logger.info(f"Connecting to Gmail for {email_address}")
        
        # Authenticate
        creds = self.authenticate()
        
        # Connect to IMAP
        self.imap_connection = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        auth_string = f'user={email_address}\x01auth=Bearer {creds.token}\x01\x01'
        self.imap_connection.authenticate('XOAUTH2', lambda x: auth_string)
        
        logger.info("Successfully connected to Gmail")

    def get_folder_list(self) -> List[str]:
        """
        Get list of available Gmail folders.
        
        Returns:
            List of folder names
        """
        if not self.imap_connection:
            raise IMAPConnectionError("Not connected to Gmail")
        
        status, folders = self.imap_connection.list()
        if status != 'OK':
            raise IMAPConnectionError(f"Failed to get folder list: {folders}")
        
        folder_names = []
        for folder in folders:
            # Parse folder name from IMAP response
            parts = folder.decode().split('"')
            if len(parts) >= 3:
                folder_name = parts[-2]
                folder_names.append(folder_name)
        
        return folder_names

    def _decode_header(self, header_value: str) -> str:
        """Decode email header value."""
        if not header_value:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(header_value):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(part.decode(encoding or 'utf-8', errors='ignore'))
                except (UnicodeDecodeError, LookupError):
                    decoded_parts.append(part.decode('utf-8', errors='ignore'))
            else:
                decoded_parts.append(part)
        
        return ''.join(decoded_parts)

    def _extract_email_body(self, email_message) -> str:
        """Extract email body content."""
        body = ""
        
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    if "attachment" in content_disposition:
                        continue
                    
                    if content_type == "text/plain":
                        charset = part.get_content_charset() or 'utf-8'
                        part_body = part.get_payload(decode=True)
                        if part_body:
                            body += part_body.decode(charset, errors='ignore')
                    elif content_type == "text/html" and not body:
                        charset = part.get_content_charset() or 'utf-8'
                        part_body = part.get_payload(decode=True)
                        if part_body:
                            # Basic HTML to text conversion
                            html_body = part_body.decode(charset, errors='ignore')
                            # Remove HTML tags
                            body = re.sub(r'<[^>]+>', '', html_body)
            else:
                charset = email_message.get_content_charset() or 'utf-8'
                payload = email_message.get_payload(decode=True)
                if payload:
                    body = payload.decode(charset, errors='ignore')
                    
        except Exception as e:
            logger.warning(f"Failed to extract email body: {e}")
            body = "Failed to extract email body"
        
        return body.strip()

    def _extract_attachments(self, email_message) -> List[str]:
        """Extract attachment filenames from email."""
        attachments = []
        
        try:
            for part in email_message.walk():
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        filename = self._decode_header(filename)
                        attachments.append(filename)
        except Exception as e:
            logger.warning(f"Failed to extract attachments: {e}")
        
        return attachments

    def _parse_email(self, raw_email: bytes, uid: str, folder: str) -> Optional[EmailData]:
        """
        Parse raw email data into EmailData object.
        
        Args:
            raw_email: Raw email bytes
            uid: Email UID
            folder: Folder name
            
        Returns:
            EmailData object or None if parsing fails
        """
        try:
            email_message = email.message_from_bytes(raw_email)
            
            # Extract headers
            message_id = email_message.get('Message-ID', f"scraped_{uid}_{int(time.time())}")
            from_address = self._decode_header(email_message.get('From', ''))
            to_address = self._decode_header(email_message.get('To', ''))
            cc_address = self._decode_header(email_message.get('Cc', '')) or None
            bcc_address = self._decode_header(email_message.get('Bcc', '')) or None
            subject = self._decode_header(email_message.get('Subject', ''))
            
            # Parse date
            date_str = email_message.get('Date')
            timestamp = datetime.now()
            if date_str:
                try:
                    timestamp = email.utils.parsedate_to_datetime(date_str)
                except Exception:
                    pass
            
            # Extract body and attachments
            body = self._extract_email_body(email_message)
            attachments = self._extract_attachments(email_message)
            
            # Determine if email is read (Gmail specific)
            # Note: This requires Gmail API for accurate read status
            is_read = False  # Default to unread, can be enhanced later
            
            # Basic categorization based on folder
            category = self._categorize_by_folder(folder)
            
            return EmailData(
                id=message_id,
                from_address=from_address,
                to_address=to_address,
                cc_address=cc_address,
                bcc_address=bcc_address,
                subject=subject,
                body=body,
                timestamp=timestamp,
                category=category,
                priority='medium',  # Default priority
                attachments=attachments,
                is_read=is_read,
                folder=folder
            )
            
        except Exception as e:
            logger.error(f"Failed to parse email {uid}: {e}")
            return None

    def _categorize_by_folder(self, folder: str) -> str:
        """Categorize email based on folder name."""
        folder_lower = folder.lower()
        
        if 'inbox' in folder_lower:
            return 'inbox'
        elif 'sent' in folder_lower:
            return 'sent'
        elif 'draft' in folder_lower:
            return 'draft'
        elif 'spam' in folder_lower or 'junk' in folder_lower:
            return 'spam'
        elif 'trash' in folder_lower or 'bin' in folder_lower:
            return 'trash'
        else:
            return 'other'

    def scrape_folder(self, folder_name: str, 
                     max_emails: Optional[int] = None,
                     incremental: bool = True) -> List[EmailData]:
        """
        Scrape emails from a specific folder.
        
        Args:
            folder_name: Name of folder to scrape
            max_emails: Maximum number of emails to fetch (None for all)
            incremental: Only fetch emails newer than last sync
            
        Returns:
            List of EmailData objects
        """
        if not self.imap_connection:
            raise IMAPConnectionError("Not connected to Gmail")
        
        logger.info(f"Scraping folder: {folder_name}")
        
        # Select folder
        status, messages = self.imap_connection.select(f'"{folder_name}"')
        if status != 'OK':
            raise IMAPConnectionError(f"Failed to select folder {folder_name}: {messages}")
        
        # Search for emails
        search_criteria = 'ALL'
        if incremental and folder_name in self.sync_state['folder_uids']:
            last_uid = self.sync_state['folder_uids'][folder_name]
            search_criteria = f'UID {last_uid + 1}:*'
        
        status, email_ids = self.imap_connection.search(None, search_criteria)
        if status != 'OK':
            logger.warning(f"Search failed for folder {folder_name}: {email_ids}")
            return []
        
        email_id_list = email_ids[0].split()
        if not email_id_list:
            logger.info(f"No emails found in folder {folder_name}")
            return []
        
        # Limit emails if specified
        if max_emails:
            email_id_list = email_id_list[-max_emails:]
        
        logger.info(f"Found {len(email_id_list)} emails in {folder_name}")
        
        emails = []
        processed = 0
        
        for email_id in email_id_list:
            try:
                # Fetch email
                status, email_data = self.imap_connection.fetch(email_id, '(RFC822 UID)')
                if status != 'OK':
                    logger.warning(f"Failed to fetch email {email_id}")
                    self.stats['errors'].append(f"Failed to fetch {email_id}")
                    continue
                
                # Extract UID and raw email
                raw_email = email_data[0][1]
                uid_data = email_data[0][0]
                uid = int(uid_data.decode().split()[-1].rstrip(')'))
                
                # Parse email
                email_obj = self._parse_email(raw_email, str(uid), folder_name)
                if email_obj:
                    emails.append(email_obj)
                    
                    # Update sync state
                    if folder_name not in self.sync_state['folder_uids']:
                        self.sync_state['folder_uids'][folder_name] = 0
                    self.sync_state['folder_uids'][folder_name] = max(
                        self.sync_state['folder_uids'][folder_name], uid
                    )
                
                processed += 1
                self.stats['total_processed'] += 1
                
                if processed % 100 == 0:
                    logger.info(f"Processed {processed}/{len(email_id_list)} emails from {folder_name}")
                
            except Exception as e:
                logger.error(f"Error processing email {email_id}: {e}")
                self.stats['errors'].append(f"Error processing {email_id}: {str(e)}")
                continue
        
        logger.info(f"Successfully scraped {len(emails)} emails from {folder_name}")
        return emails

    def scrape_all_folders(self, 
                          folders: Optional[List[str]] = None,
                          max_emails_per_folder: Optional[int] = None,
                          incremental: bool = True) -> List[EmailData]:
        """
        Scrape emails from multiple folders.
        
        Args:
            folders: List of folder names (None for all folders)
            max_emails_per_folder: Max emails per folder
            incremental: Only fetch new emails
            
        Returns:
            List of all scraped EmailData objects
        """
        self.stats['start_time'] = datetime.now()
        
        if folders is None:
            folders = self.get_folder_list()
        
        # Filter out folders that might cause issues
        excluded_folders = {'[Gmail]/All Mail', '[Gmail]/Spam', '[Gmail]/Trash'}
        folders = [f for f in folders if f not in excluded_folders]
        
        logger.info(f"Scraping {len(folders)} folders: {folders}")
        self.stats['total_folders'] = len(folders)
        
        all_emails = []
        
        for folder in folders:
            try:
                folder_emails = self.scrape_folder(
                    folder, 
                    max_emails_per_folder, 
                    incremental
                )
                all_emails.extend(folder_emails)
                self.stats['total_new'] += len(folder_emails)
                
                logger.info(f"Total emails collected so far: {len(all_emails)}")
                
            except Exception as e:
                logger.error(f"Failed to scrape folder {folder}: {e}")
                self.stats['errors'].append(f"Failed to scrape {folder}: {str(e)}")
                continue
        
        self.stats['end_time'] = datetime.now()
        self.scraped_emails = all_emails
        
        # Update sync state
        self.sync_state['last_sync'] = datetime.now().isoformat()
        self.sync_state['total_emails'] = len(all_emails)
        self._save_sync_state()
        
        logger.info(f"Scraping completed! Total emails: {len(all_emails)}")
        return all_emails

    def save_emails(self, 
                   emails: Optional[List[EmailData]] = None,
                   filename: Optional[str] = None,
                   format: str = 'json') -> str:
        """
        Save emails to file.
        
        Args:
            emails: List of emails to save (None for scraped emails)
            filename: Output filename (None for auto-generated)
            format: Output format ('json' or 'csv')
            
        Returns:
            Path to saved file
        """
        if emails is None:
            emails = self.scraped_emails
        
        if not emails:
            raise ValueError("No emails to save")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if filename is None:
            filename = f"gmail_emails_{timestamp}.{format}"
        
        file_path = self.data_dir / filename
        
        logger.info(f"Saving {len(emails)} emails to {file_path}")
        
        if format.lower() == 'json':
            email_dicts = [asdict(email) for email in emails]
            # Convert datetime objects to ISO strings
            for email_dict in email_dicts:
                if email_dict['timestamp']:
                    email_dict['timestamp'] = email_dict['timestamp'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(email_dicts, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == 'csv':
            # Use EmailDataLoader export functionality
            loader = EmailDataLoader()
            loader.export_to_csv(emails, file_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Successfully saved emails to {file_path}")
        return str(file_path)

    def get_stats(self) -> Dict:
        """Get scraping statistics."""
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            duration = stats['end_time'] - stats['start_time']
            stats['duration_seconds'] = duration.total_seconds()
            stats['emails_per_second'] = (
                stats['total_processed'] / duration.total_seconds() 
                if duration.total_seconds() > 0 else 0
            )
        
        stats['sync_state'] = self.sync_state.copy()
        return stats

    def disconnect(self):
        """Disconnect from Gmail."""
        if self.imap_connection:
            try:
                self.imap_connection.close()
                self.imap_connection.logout()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.imap_connection = None
        
        logger.info("Disconnected from Gmail")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def main():
    """Example usage of EmailScraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gmail Email Scraper')
    parser.add_argument('email', help='Gmail address to scrape')
    parser.add_argument('--folders', nargs='*', help='Specific folders to scrape')
    parser.add_argument('--max-emails', type=int, help='Max emails per folder')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--full-sync', action='store_true', help='Full sync (ignore incremental)')
    parser.add_argument('--output', help='Output filename')
    
    args = parser.parse_args()
    
    scraper = EmailScraper()
    
    try:
        # Connect to Gmail
        scraper.connect_gmail(args.email)
        
        # Scrape emails
        emails = scraper.scrape_all_folders(
            folders=args.folders,
            max_emails_per_folder=args.max_emails,
            incremental=not args.full_sync
        )
        
        # Save emails
        if emails:
            file_path = scraper.save_emails(
                emails,
                filename=args.output,
                format=args.format
            )
            
            # Print statistics
            stats = scraper.get_stats()
            print(f"\nScraping completed!")
            print(f"Total emails scraped: {stats['total_new']}")
            print(f"Total emails processed: {stats['total_processed']}")
            print(f"Folders scraped: {stats['total_folders']}")
            print(f"Saved to: {file_path}")
            
            if stats.get('duration_seconds'):
                print(f"Duration: {stats['duration_seconds']:.1f} seconds")
                print(f"Speed: {stats.get('emails_per_second', 0):.1f} emails/second")
            
            if stats['errors']:
                print(f"Errors: {len(stats['errors'])}")
        else:
            print("No emails found to scrape")
    
    finally:
        scraper.disconnect()


if __name__ == "__main__":
    main()