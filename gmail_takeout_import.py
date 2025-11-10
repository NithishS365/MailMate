#!/usr/bin/env python3
"""
Gmail Takeout Import Tool

Import Gmail data from Google Takeout archive into MailMate format.
This bypasses network connectivity issues by using manually downloaded Gmail data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import email
import mailbox
import re

# Add backend to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

def main():
    """Import Gmail data from Google Takeout."""
    print("📧 MailMate Gmail Takeout Import Tool")
    print("=" * 50)
    print("Import your Gmail data from Google Takeout archive")
    print()
    print("📋 Steps to get your data:")
    print("1. Go to https://takeout.google.com")
    print("2. Select 'Gmail' only")
    print("3. Choose format: 'JSON' (recommended) or 'MBOX'") 
    print("4. Download the archive")
    print("5. Extract it and point this tool to the Gmail folder")
    print()
    
    # Get archive path
    archive_path = input("📁 Enter path to extracted Gmail folder: ").strip().strip('"')
    if not archive_path or not Path(archive_path).exists():
        print("❌ Invalid path or folder doesn't exist")
        return
    
    archive_path = Path(archive_path)
    
    # Ask for limit
    limit_input = input("📊 How many emails to import? (default: all): ").strip()
    limit = int(limit_input) if limit_input.isdigit() else None
    
    # Data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        print(f"\n🚀 Starting Gmail Takeout import...")
        print(f"📂 Source: {archive_path}")
        print(f"💾 Output: {data_dir}")
        
        # Check for different file formats
        json_files = list(archive_path.glob("*.json"))
        mbox_files = list(archive_path.glob("*.mbox"))
        
        emails_data = []
        
        if json_files:
            print(f"\n📄 Found {len(json_files)} JSON files")
            emails_data = import_from_json(json_files, limit)
        elif mbox_files:
            print(f"\n📄 Found {len(mbox_files)} MBOX files")  
            emails_data = import_from_mbox(mbox_files, limit)
        else:
            print("❌ No supported email files found (JSON or MBOX)")
            print("   Make sure you're pointing to the Gmail folder from the extracted archive")
            return
        
        if not emails_data:
            print("⚠️  No emails were imported")
            return
        
        # Save emails
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = data_dir / f"gmail_takeout_{timestamp}.json"
        
        print(f"\n💾 Saving {len(emails_data)} emails to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(emails_data, f, indent=2, ensure_ascii=False)
        
        # Show results
        print("\n" + "=" * 50)
        print("✅ Gmail import completed successfully!")
        print(f"📊 Results:")
        print(f"   • Emails imported: {len(emails_data)}")
        print(f"   • Output file: {output_file}")
        
        print(f"\n🎉 Your Gmail data is now ready to use!")
        print(f"\n🔄 To use this data in MailMate:")
        print(f"   1. Restart the MailMate server (if running)")
        print(f"   2. The server will automatically load your imported emails")
        print(f"   3. Visit http://localhost:5000 to view your dashboard")
        
    except Exception as e:
        print(f"\n❌ Error during import: {e}")
        print(f"Error type: {type(e).__name__}")

def import_from_json(json_files, limit=None):
    """Import emails from JSON files."""
    emails_data = []
    count = 0
    
    for json_file in json_files:
        print(f"   📄 Processing {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures from Google Takeout
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict) and 'messages' in data:
                messages = data['messages']
            else:
                print(f"   ⚠️  Unknown JSON structure in {json_file.name}")
                continue
            
            for msg in messages:
                if limit and count >= limit:
                    break
                
                email_data = convert_gmail_message(msg)
                if email_data:
                    emails_data.append(email_data)
                    count += 1
            
            if limit and count >= limit:
                break
                
        except Exception as e:
            print(f"   ❌ Error processing {json_file.name}: {e}")
    
    return emails_data

def import_from_mbox(mbox_files, limit=None):
    """Import emails from MBOX files."""
    emails_data = []
    count = 0
    
    for mbox_file in mbox_files:
        print(f"   📄 Processing {mbox_file.name}...")
        
        try:
            mbox = mailbox.mbox(str(mbox_file))
            
            for message in mbox:
                if limit and count >= limit:
                    break
                
                email_data = convert_mbox_message(message)
                if email_data:
                    emails_data.append(email_data)
                    count += 1
            
            if limit and count >= limit:
                break
                
        except Exception as e:
            print(f"   ❌ Error processing {mbox_file.name}: {e}")
    
    return emails_data

def convert_gmail_message(msg):
    """Convert Gmail API message to MailMate format."""
    try:
        return {
            'id': msg.get('id', f"takeout_{hash(str(msg))}"),
            'from_address': extract_email(msg.get('from', '')),
            'to_address': extract_email(msg.get('to', '')),
            'cc_address': extract_email(msg.get('cc', '')),
            'subject': msg.get('subject', ''),
            'body': msg.get('body', msg.get('snippet', '')),
            'timestamp': parse_date(msg.get('date')),
            'category': determine_category(msg.get('labels', [])),
            'priority': determine_priority(msg),
            'attachments': extract_attachments(msg),
            'is_read': not ('UNREAD' in msg.get('labels', [])),
            'folder': determine_folder(msg.get('labels', []))
        }
    except Exception as e:
        print(f"   ⚠️  Error converting message: {e}")
        return None

def convert_mbox_message(message):
    """Convert MBOX message to MailMate format."""
    try:
        return {
            'id': message.get('Message-ID', f"mbox_{hash(str(message))}"),
            'from_address': extract_email(message.get('From', '')),
            'to_address': extract_email(message.get('To', '')),
            'cc_address': extract_email(message.get('Cc', '')),
            'subject': message.get('Subject', ''),
            'body': get_message_body(message),
            'timestamp': parse_date(message.get('Date')),
            'category': 'inbox',
            'priority': 'medium',
            'attachments': [],
            'is_read': True,
            'folder': 'INBOX'
        }
    except Exception as e:
        print(f"   ⚠️  Error converting MBOX message: {e}")
        return None

def extract_email(email_str):
    """Extract email address from string."""
    if not email_str:
        return ''
    
    # Use regex to extract email
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_str)
    return match.group(0) if match else email_str

def parse_date(date_str):
    """Parse date string to ISO format."""
    if not date_str:
        return None
    
    try:
        # Try different date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%d %b %Y %H:%M:%S %z', 
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
                
        # If all fails, return current time
        return datetime.now().isoformat()
        
    except Exception:
        return datetime.now().isoformat()

def determine_category(labels):
    """Determine email category from labels."""
    labels_str = ' '.join(labels).lower()
    
    if 'important' in labels_str:
        return 'important'
    elif 'sent' in labels_str:
        return 'sent'
    elif 'draft' in labels_str:
        return 'draft'
    elif 'spam' in labels_str:
        return 'spam'
    else:
        return 'inbox'

def determine_priority(msg):
    """Determine email priority."""
    subject = msg.get('subject', '').lower()
    
    if any(word in subject for word in ['urgent', 'important', 'asap']):
        return 'high'
    elif any(word in subject for word in ['fyi', 'info', 'newsletter']):
        return 'low'
    else:
        return 'medium'

def determine_folder(labels):
    """Determine folder from labels."""
    if 'SENT' in labels:
        return 'Sent'
    elif 'DRAFT' in labels:
        return 'Drafts'
    elif 'SPAM' in labels:
        return 'Spam'
    else:
        return 'INBOX'

def extract_attachments(msg):
    """Extract attachment names."""
    attachments = []
    if 'attachments' in msg:
        for att in msg['attachments']:
            if 'filename' in att:
                attachments.append(att['filename'])
    return attachments

def get_message_body(message):
    """Extract body text from email message."""
    body = ""
    
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
    
    return body[:5000]  # Limit body size

if __name__ == "__main__":
    main()