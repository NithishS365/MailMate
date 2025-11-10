#!/usr/bin/env python3
"""
Quick Email Scraper Script for MailMate

Simple script to scrape Gmail emails and store them for use in MailMate.
This script provides an easy way to get started with real email data.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

def main():
    """Main function to run email scraping."""
    from backend.email_scraper import EmailScraper
    
    print("🎯 MailMate Gmail Scraper")
    print("=" * 50)
    
    # Get email address
    email = input("📧 Enter your Gmail address: ").strip()
    if not email:
        print("❌ Email address is required")
        return
    
    # Ask for scraping options
    print("\n🔧 Scraping Options:")
    print("1. Quick sync (INBOX only, last 100 emails)")
    print("2. Standard sync (INBOX + Sent, last 500 emails)")  
    print("3. Full sync (all folders, all emails)")
    print("4. Custom sync")
    
    choice = input("\nSelect option (1-4) [1]: ").strip() or "1"
    
    # Set up scraping parameters
    if choice == "1":
        folders = ["INBOX"]
        max_emails = 100
        print("📋 Quick sync: INBOX, 100 emails")
    elif choice == "2":
        folders = ["INBOX", "[Gmail]/Sent Mail"]
        max_emails = 500
        print("📋 Standard sync: INBOX + Sent, 500 emails each")
    elif choice == "3":
        folders = None
        max_emails = None
        print("📋 Full sync: All folders, all emails")
    else:
        # Custom options
        folder_input = input("📁 Folders (comma-separated) [INBOX]: ").strip()
        folders = [f.strip() for f in folder_input.split(",")] if folder_input else ["INBOX"]
        
        max_input = input("📊 Max emails per folder (empty for all): ").strip()
        max_emails = int(max_input) if max_input.isdigit() else None
        
        print(f"📋 Custom sync: {folders}, {max_emails or 'all'} emails each")
    
    # Data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        print(f"\n🚀 Starting email scraping...")
        print(f"📂 Data will be saved to: {data_dir}")
        
        with EmailScraper(data_dir=str(data_dir)) as scraper:
            # Connect to Gmail
            print("\n🔐 Connecting to Gmail...")
            print("   (This will open a browser window for OAuth authentication)")
            scraper.connect_gmail(email)
            print("✅ Connected successfully!")
            
            # Show available folders if doing full sync
            if folders is None:
                available_folders = scraper.get_folder_list()
                print(f"\n📋 Available folders ({len(available_folders)}):")
                for folder in available_folders[:10]:  # Show first 10
                    print(f"   • {folder}")
                if len(available_folders) > 10:
                    print(f"   • ... and {len(available_folders) - 10} more")
            
            # Scrape emails
            print(f"\n📥 Scraping emails...")
            emails = scraper.scrape_all_folders(
                folders=folders,
                max_emails_per_folder=max_emails,
                incremental=True
            )
            
            if not emails:
                print("⚠️  No emails found to scrape")
                return
            
            # Save emails
            print(f"\n💾 Saving {len(emails)} emails...")
            file_path = scraper.save_emails(emails, format='json')
            
            # Show results
            stats = scraper.get_stats()
            print("\n" + "=" * 50)
            print("✅ Scraping completed successfully!")
            print(f"📊 Results:")
            print(f"   • Emails scraped: {stats['total_new']}")
            print(f"   • Folders processed: {stats['total_folders']}")
            print(f"   • Output file: {file_path}")
            
            if stats.get('duration_seconds'):
                print(f"   • Time taken: {stats['duration_seconds']:.1f} seconds")
                print(f"   • Speed: {stats.get('emails_per_second', 0):.1f} emails/sec")
            
            print(f"\n🎉 Your emails are now saved and ready to use!")
            print(f"📁 File location: {file_path}")
            print(f"\n🔄 To use this data in MailMate:")
            print(f"   1. Start the MailMate server: python mailmate_server.py")
            print(f"   2. The server will automatically load your scraped emails")
            print(f"   3. Visit http://localhost:5000 to view your dashboard")
            
            if stats['errors']:
                print(f"\n⚠️  Encountered {len(stats['errors'])} errors during scraping")
                print("   Check the logs for details")
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Scraping cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure you have a valid Gmail account")
        print("   • Check your internet connection")
        print("   • Ensure OAuth2 credentials are set up correctly")
        print("   • Run with --verbose for more detailed error information")


if __name__ == "__main__":
    main()