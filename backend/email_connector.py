#!/usr/bin/env python3
"""
MailMate Email Connectivity Module

This module provides real email connectivity for Gmail, Outlook, Yahoo, and other providers
using IMAP, POP3, and OAuth2 authentication methods.
"""

import imaplib
import poplib
import smtplib
import ssl
import email
import json
import base64
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import requests

# Set up logging
logger = logging.getLogger(__name__)

class EmailConnectionError(Exception):
    """Custom exception for email connection issues."""
    pass

class EmailAuthenticationError(Exception):
    """Custom exception for email authentication issues."""
    pass

class EmailProvider:
    """Base class for email providers with common configurations."""
    
    # Common email provider configurations
    PROVIDERS = {
        'gmail': {
            'imap_server': 'imap.gmail.com',
            'imap_port': 993,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'oauth2_enabled': True,
            'requires_app_password': True
        },
        'outlook': {
            'imap_server': 'outlook.office365.com',
            'imap_port': 993,
            'smtp_server': 'smtp-mail.outlook.com',
            'smtp_port': 587,
            'oauth2_enabled': True,
            'requires_app_password': False
        },
        'yahoo': {
            'imap_server': 'imap.mail.yahoo.com',
            'imap_port': 993,
            'smtp_server': 'smtp.mail.yahoo.com',
            'smtp_port': 587,
            'oauth2_enabled': False,
            'requires_app_password': True
        },
        'icloud': {
            'imap_server': 'imap.mail.me.com',
            'imap_port': 993,
            'smtp_server': 'smtp.mail.me.com',
            'smtp_port': 587,
            'oauth2_enabled': False,
            'requires_app_password': True
        },
        'generic_imap': {
            'imap_server': None,  # Will be auto-detected
            'imap_port': 993,
            'smtp_server': None,
            'smtp_port': 587,
            'oauth2_enabled': False,
            'requires_app_password': False
        },
        'custom': {
            'imap_server': None,
            'imap_port': 993,
            'smtp_server': None,
            'smtp_port': 587,
            'oauth2_enabled': False,
            'requires_app_password': False
        }
    }

class EmailConnector:
    """
    Main email connector class that handles connections to various email providers.
    """
    
    def __init__(self):
        self.connections = {}
        self.logger = logging.getLogger(f"{__name__}.EmailConnector")
        
    def connect_imap(self, 
                    email_address: str, 
                    password: str, 
                    provider: str = 'auto',
                    custom_server: Optional[str] = None,
                    custom_port: Optional[int] = None) -> Dict[str, Any]:
        """
        Connect to email account using IMAP.
        
        Args:
            email_address: User's email address
            password: Email password or app-specific password
            provider: Email provider ('gmail', 'outlook', 'yahoo', 'icloud', 'custom')
            custom_server: Custom IMAP server (for 'custom' provider)
            custom_port: Custom IMAP port (for 'custom' provider)
            
        Returns:
            Dictionary with connection status and details
        """
        try:
            # Auto-detect provider if needed
            if provider == 'auto':
                provider = self._detect_provider(email_address)
            
            # Get provider configuration
            if provider not in EmailProvider.PROVIDERS:
                raise EmailConnectionError(f"Unsupported provider: {provider}")
            
            config = EmailProvider.PROVIDERS[provider].copy()
            
            # Use custom server settings if provided
            if provider == 'custom':
                if not custom_server:
                    raise EmailConnectionError("Custom server required for custom provider")
                config['imap_server'] = custom_server
                if custom_port:
                    config['imap_port'] = custom_port
            elif provider == 'generic_imap':
                # Auto-detect IMAP server for generic domains
                domain = email_address.split('@')[1]
                config['imap_server'] = self._auto_detect_imap_server(domain)
                if not config['imap_server']:
                    raise EmailConnectionError(
                        f"Could not auto-detect IMAP server for {domain}. "
                        f"Please use 'Custom' provider and enter server details manually."
                    )
            
            # Create IMAP connection
            imap_server = config['imap_server']
            imap_port = config['imap_port']
            
            self.logger.info(f"Connecting to {imap_server}:{imap_port} for {email_address}")
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect to server
            mail = imaplib.IMAP4_SSL(imap_server, imap_port, ssl_context=context)
            
            # Login
            mail.login(email_address, password)
            
            # Get mailbox info
            mailboxes = self._get_mailboxes(mail)
            
            # Store connection
            connection_id = f"{email_address}_{datetime.now().timestamp():.0f}"
            self.connections[connection_id] = {
                'connection': mail,
                'email': email_address,
                'provider': provider,
                'type': 'imap',
                'connected_at': datetime.now(),
                'mailboxes': mailboxes
            }
            
            self.logger.info(f"Successfully connected to {email_address} via IMAP")
            
            return {
                'status': 'success',
                'connection_id': connection_id,
                'email': email_address,
                'provider': provider,
                'mailboxes': mailboxes,
                'message': f'Successfully connected to {email_address}'
            }
            
        except imaplib.IMAP4.error as e:
            error_msg = f"IMAP authentication failed: {str(e)}"
            self.logger.error(error_msg)
            raise EmailAuthenticationError(error_msg)
        except Exception as e:
            error_msg = f"Failed to connect to {email_address}: {str(e)}"
            self.logger.error(error_msg)
            raise EmailConnectionError(error_msg)
    
    def fetch_emails(self, 
                    connection_id: str, 
                    mailbox: str = 'INBOX',
                    limit: int = 50,
                    since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch emails from connected account.
        
        Args:
            connection_id: Connection identifier
            mailbox: Mailbox to fetch from (default: INBOX)
            limit: Maximum number of emails to fetch
            since_date: Only fetch emails after this date
            
        Returns:
            List of email dictionaries
        """
        try:
            if connection_id not in self.connections:
                raise EmailConnectionError(f"Connection {connection_id} not found")
            
            conn_info = self.connections[connection_id]
            mail = conn_info['connection']
            
            # Select mailbox
            mail.select(mailbox)
            
            # Build search criteria
            search_criteria = 'ALL'
            if since_date:
                date_str = since_date.strftime('%d-%b-%Y')
                search_criteria = f'SINCE {date_str}'
            
            # Search for emails
            status, messages = mail.search(None, search_criteria)
            if status != 'OK':
                raise EmailConnectionError(f"Failed to search emails: {status}")
            
            # Get message IDs
            message_ids = messages[0].split()
            
            # Limit results
            if limit and len(message_ids) > limit:
                message_ids = message_ids[-limit:]  # Get most recent
            
            emails = []
            for msg_id in message_ids:
                try:
                    # Fetch email
                    status, msg_data = mail.fetch(msg_id, '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    # Parse email
                    email_obj = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email information
                    email_info = self._parse_email(email_obj, msg_id.decode())
                    emails.append(email_info)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse email {msg_id}: {e}")
                    continue
            
            self.logger.info(f"Fetched {len(emails)} emails from {conn_info['email']}")
            return emails
            
        except Exception as e:
            error_msg = f"Failed to fetch emails: {str(e)}"
            self.logger.error(error_msg)
            raise EmailConnectionError(error_msg)
    
    def get_mailbox_stats(self, connection_id: str) -> Dict[str, Any]:
        """
        Get statistics for all mailboxes in the account.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Dictionary with mailbox statistics
        """
        try:
            if connection_id not in self.connections:
                raise EmailConnectionError(f"Connection {connection_id} not found")
            
            conn_info = self.connections[connection_id]
            mail = conn_info['connection']
            
            stats = {}
            for mailbox in conn_info['mailboxes']:
                try:
                    mail.select(mailbox, readonly=True)
                    status, messages = mail.search(None, 'ALL')
                    
                    if status == 'OK':
                        message_count = len(messages[0].split()) if messages[0] else 0
                        
                        # Get unread count
                        status, unread = mail.search(None, 'UNSEEN')
                        unread_count = len(unread[0].split()) if unread[0] else 0
                        
                        stats[mailbox] = {
                            'total_messages': message_count,
                            'unread_messages': unread_count,
                            'name': mailbox
                        }
                    else:
                        stats[mailbox] = {
                            'total_messages': 0,
                            'unread_messages': 0,
                            'name': mailbox,
                            'error': 'Failed to access mailbox'
                        }
                        
                except Exception as e:
                    stats[mailbox] = {
                        'total_messages': 0,
                        'unread_messages': 0,
                        'name': mailbox,
                        'error': str(e)
                    }
            
            return {
                'email': conn_info['email'],
                'provider': conn_info['provider'],
                'connected_at': conn_info['connected_at'].isoformat(),
                'mailboxes': stats
            }
            
        except Exception as e:
            error_msg = f"Failed to get mailbox stats: {str(e)}"
            self.logger.error(error_msg)
            raise EmailConnectionError(error_msg)
    
    def disconnect(self, connection_id: str) -> Dict[str, str]:
        """
        Disconnect from email account.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Disconnection status
        """
        try:
            if connection_id not in self.connections:
                return {'status': 'error', 'message': 'Connection not found'}
            
            conn_info = self.connections[connection_id]
            
            # Close connection
            if conn_info['type'] == 'imap':
                conn_info['connection'].close()
                conn_info['connection'].logout()
            
            # Remove from connections
            del self.connections[connection_id]
            
            self.logger.info(f"Disconnected from {conn_info['email']}")
            
            return {
                'status': 'success',
                'message': f"Disconnected from {conn_info['email']}"
            }
            
        except Exception as e:
            error_msg = f"Failed to disconnect: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """
        List all active connections.
        
        Returns:
            List of connection information
        """
        connections = []
        for conn_id, conn_info in self.connections.items():
            connections.append({
                'connection_id': conn_id,
                'email': conn_info['email'],
                'provider': conn_info['provider'],
                'type': conn_info['type'],
                'connected_at': conn_info['connected_at'].isoformat(),
                'mailbox_count': len(conn_info.get('mailboxes', []))
            })
        
        return connections
    
    def _auto_detect_imap_server(self, domain: str) -> Optional[str]:
        """
        Auto-detect IMAP server for a domain by trying common patterns.
        
        Args:
            domain: Email domain (e.g., 'sece.ac.in')
            
        Returns:
            IMAP server address if found, None otherwise
        """
        # Common IMAP server patterns
        patterns = [
            f'imap.{domain}',
            f'mail.{domain}',
            f'{domain}',
            f'imap.mail.{domain}',
            f'secure.{domain}',
            f'pop.{domain}'  # Some servers use pop prefix for IMAP too
        ]
        
        import socket
        
        for pattern in patterns:
            try:
                # Try to resolve the hostname
                socket.gethostbyname(pattern)
                
                # Try to connect to port 993 (IMAP SSL)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)  # 5 second timeout
                result = sock.connect_ex((pattern, 993))
                sock.close()
                
                if result == 0:  # Connection successful
                    self.logger.info(f"Auto-detected IMAP server: {pattern}")
                    return pattern
                    
            except Exception as e:
                # Continue to next pattern
                continue
        
        self.logger.warning(f"Could not auto-detect IMAP server for domain: {domain}")
        return None
    
    def _detect_provider(self, email_address: str) -> str:
        """Auto-detect email provider from email address."""
        domain = email_address.split('@')[1].lower()
        
        if 'gmail.com' in domain:
            return 'gmail'
        elif 'outlook.com' in domain or 'hotmail.com' in domain or 'live.com' in domain:
            return 'outlook'
        elif 'yahoo.com' in domain:
            return 'yahoo'
        elif 'icloud.com' in domain or 'me.com' in domain:
            return 'icloud'
        else:
            # For unknown domains, try common IMAP configurations
            # Most educational and corporate emails use standard IMAP settings
            return 'generic_imap'
    
    def _get_mailboxes(self, mail: imaplib.IMAP4_SSL) -> List[str]:
        """Get list of available mailboxes."""
        try:
            status, mailboxes = mail.list()
            if status != 'OK':
                return ['INBOX']
            
            mailbox_names = []
            for mailbox in mailboxes:
                # Parse mailbox name
                parts = mailbox.decode().split('" "')
                if len(parts) >= 2:
                    name = parts[-1].strip('"')
                    mailbox_names.append(name)
            
            return mailbox_names if mailbox_names else ['INBOX']
            
        except Exception as e:
            self.logger.warning(f"Failed to get mailboxes: {e}")
            return ['INBOX']
    
    def _parse_email(self, email_obj: email.message.Message, msg_id: str) -> Dict[str, Any]:
        """Parse email object into dictionary."""
        
        def decode_mime_words(s):
            """Decode MIME encoded words."""
            if not s:
                return ""
            try:
                decoded_parts = decode_header(s)
                result = ""
                for part, encoding in decoded_parts:
                    if isinstance(part, bytes):
                        if encoding:
                            result += part.decode(encoding)
                        else:
                            result += part.decode('utf-8', errors='ignore')
                    else:
                        result += part
                return result
            except:
                return s
        
        # Extract basic headers
        subject = decode_mime_words(email_obj.get('Subject', ''))
        from_addr = decode_mime_words(email_obj.get('From', ''))
        to_addr = decode_mime_words(email_obj.get('To', ''))
        date_str = email_obj.get('Date', '')
        
        # Parse date
        try:
            date_obj = email.utils.parsedate_to_datetime(date_str)
            timestamp = date_obj.isoformat()
        except:
            timestamp = datetime.now().isoformat()
        
        # Extract body
        body = ""
        html_body = ""
        
        if email_obj.is_multipart():
            for part in email_obj.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='ignore')
                elif content_type == "text/html":
                    charset = part.get_content_charset() or 'utf-8'
                    html_body = part.get_payload(decode=True).decode(charset, errors='ignore')
        else:
            charset = email_obj.get_content_charset() or 'utf-8'
            body = email_obj.get_payload(decode=True).decode(charset, errors='ignore')
        
        # Determine category (basic classification)
        category = self._classify_email(subject, from_addr, body)
        
        return {
            'id': msg_id,
            'subject': subject,
            'from': from_addr,
            'to': to_addr,
            'date': timestamp,
            'body': body,
            'html_body': html_body,
            'category': category,
            'is_read': True,  # Assume read for now
            'has_attachments': len([p for p in email_obj.walk() if p.get_filename()]) > 0
        }
    
    def _classify_email(self, subject: str, from_addr: str, body: str) -> str:
        """Basic email classification."""
        subject_lower = subject.lower()
        from_lower = from_addr.lower()
        body_lower = body.lower()
        
        # Newsletter indicators
        if any(word in subject_lower for word in ['newsletter', 'unsubscribe', 'digest']):
            return 'Newsletter'
        if any(word in from_lower for word in ['noreply', 'no-reply', 'newsletter']):
            return 'Newsletter'
        
        # Promotional indicators
        if any(word in subject_lower for word in ['sale', 'offer', 'discount', 'deal', 'promotion']):
            return 'Promotional'
        
        # Spam indicators
        if any(word in subject_lower for word in ['urgent', 'act now', 'limited time', 'free']):
            return 'Spam'
        
        # Business indicators
        if any(word in subject_lower for word in ['meeting', 'invoice', 'proposal', 'contract']):
            return 'Business'
        
        return 'Personal'

# Global email connector instance
email_connector = EmailConnector()

def get_email_connector() -> EmailConnector:
    """Get the global email connector instance."""
    return email_connector