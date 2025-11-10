#!/usr/bin/env python3
"""
Alternative Gmail Scraper using IMAP (fallback for API issues)

This script uses IMAP with OAuth2 to fetch emails when the Gmail API has connectivity issues.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

def main():
    """Main function to run IMAP email scraping."""
    from backend.email_loader import EmailDataLoader
    
    print("📧 MailMate IMAP Gmail Scraper (Alternative Method)")
    print("=" * 60)
    print("This uses IMAP instead of Gmail API for better network compatibility")
    
    # Get email address
    email = input("\n📧 Enter your Gmail address: ").strip()
    if not email:
        print("❌ Email address is required")
        return
    
    # Ask for limit
    limit_input = input("📊 How many emails to fetch? (default: 100): ").strip()
    limit = int(limit_input) if limit_input.isdigit() else 100
    
    # Data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        print(f"\n🚀 Starting IMAP email scraping...")
        print(f"📂 Data will be saved to: {data_dir}")
        print(f"📊 Fetching last {limit} emails from INBOX")
        
        # Initialize loader
        loader = EmailDataLoader()
        
        print(f"\n🔐 Connecting to Gmail via IMAP...")
        print("   (This will use OAuth2 authentication)")
        
        # Connect using OAuth2 IMAP
        loader.connect_imap(
            email_address=email,
            password=None,  # Use OAuth2 instead
            provider='gmail',
            use_oauth2=True,
            client_secret_file='config/client_secret.json',
            token_file='config/gmail_token.pickle'
        )
        print("✅ Connected successfully!")
        
        print(f"\n📥 Fetching emails from INBOX...")
        emails = loader.fetch_emails_imap(
            folder='INBOX',
            limit=limit,
            search_criteria='ALL'
        )
        
        if not emails:
            print("⚠️  No emails found")
            return
        
        # Save emails
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = data_dir / f"gmail_imap_emails_{timestamp}.json"
        
        print(f"\n💾 Saving {len(emails)} emails to {output_file}...")
        
        # Convert to JSON format
        emails_data = []
        for email_data in emails:
            emails_data.append({
                'id': email_data.id,
                'from_address': email_data.from_address,
                'to_address': email_data.to_address,
                'subject': email_data.subject,
                'body': email_data.body,
                'timestamp': email_data.timestamp.isoformat() if email_data.timestamp else None,
                'category': email_data.category,
                'priority': email_data.priority,
                'attachments': email_data.attachments or [],
                'is_read': getattr(email_data, 'is_read', False),
                'folder': 'INBOX'
            })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(emails_data, f, indent=2, ensure_ascii=False)
        
        # Disconnect
        loader.disconnect_imap()
        
        # Show results
        print("\n" + "=" * 60)
        print("✅ IMAP scraping completed successfully!")
        print(f"📊 Results:")
        print(f"   • Emails scraped: {len(emails)}")
        print(f"   • Output file: {output_file}")
        
        print(f"\n🎉 Your emails are now saved and ready to use!")
        print(f"📁 File location: {output_file}")
        print(f"\n🔄 To use this data in MailMate:")
        print(f"   1. Restart the MailMate server (if running)")
        print(f"   2. The server will automatically load your scraped emails")
        print(f"   3. Visit http://localhost:5000 to view your dashboard")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Scraping cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        print(f"\n🔧 Error details: {type(e).__name__}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure Gmail IMAP is enabled in your account")
        print("   • Check that OAuth2 credentials are set up correctly")
        print("   • Ensure 2-factor authentication allows app access")
        print("   • Try enabling 'Less secure app access' temporarily")

if __name__ == "__main__":
    main()