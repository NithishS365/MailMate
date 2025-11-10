# Gmail Email Scraping Guide

This guide shows you how to scrape emails from your Gmail account and use them in MailMate instead of synthetic data.

## Quick Start

### Method 1: Simple Script (Recommended)

Run the simple scraping script:

```powershell
python scrape_emails.py
```

This interactive script will:

- Ask for your Gmail address
- Let you choose scraping options (quick, standard, full, or custom)
- Handle OAuth2 authentication
- Save emails to `data/` folder
- Show progress and results

### Method 2: Command Line Interface

Use the CLI for more control:

```powershell
# Scrape INBOX with last 100 emails
python backend/cli.py scrape your.email@gmail.com --folder INBOX --max 100

# Scrape multiple folders
python backend/cli.py scrape your.email@gmail.com --folder INBOX --folder "Sent" --max 500

# Full sync (all folders, all emails)
python backend/cli.py scrape your.email@gmail.com --full --output my_gmail_backup.json

# Quick sync of specific folder
python backend/cli.py sync your.email@gmail.com --folder INBOX --limit 50
```

### Method 3: Direct Python Script

```python
from backend.email_scraper import EmailScraper

with EmailScraper() as scraper:
    scraper.connect_gmail('your.email@gmail.com')
    emails = scraper.scrape_all_folders(max_emails_per_folder=100)
    file_path = scraper.save_emails(emails)
    print(f"Saved {len(emails)} emails to {file_path}")
```

## Setup Requirements

### 1. OAuth2 Configuration

You need to set up Gmail OAuth2 credentials:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API
4. Create OAuth2 credentials (Desktop application)
5. Download the credentials file as `config/client_secret.json`

### 2. Required Dependencies

Make sure you have the required packages:

```powershell
pip install google-auth google-auth-oauthlib google-auth-httplib2
```

## Scraping Options

### Folder Selection

- **INBOX**: Your main inbox
- **[Gmail]/Sent Mail**: Sent emails
- **[Gmail]/Drafts**: Draft emails
- **[Gmail]/All Mail**: All emails (not recommended, very large)
- **Custom folders**: Any labels you've created

### Sync Types

- **Incremental (default)**: Only fetch emails newer than last sync
- **Full sync**: Download all emails (can be very slow for large mailboxes)

### Output Formats

- **JSON** (recommended): Preserves all data structures
- **CSV**: Good for analysis, some data limitations

## File Structure

After scraping, your data directory will look like:

```
data/
├── gmail_emails_20251110_143022.json    # Scraped emails
├── sync_state.json                      # Sync tracking
└── gmail_token.pickle                   # OAuth tokens (keep secure!)
```

## Using Scraped Data

### Automatic Loading

MailMate automatically detects and loads scraped emails:

1. Start the server: `python mailmate_server.py`
2. The server will automatically find the latest scraped email file
3. Your real emails will appear in the dashboard

### Manual Loading

You can also load specific files:

```python
from backend.email_loader import EmailDataLoader

loader = EmailDataLoader()
emails = loader.load_scraped_emails('data/gmail_emails_20251110_143022.json')
print(f"Loaded {len(emails)} emails")
```

## Advanced Usage

### Incremental Sync

MailMate tracks what emails have been downloaded and only fetches new ones:

```powershell
# First run - downloads all emails
python scrape_emails.py

# Later runs - only downloads new emails
python scrape_emails.py
```

The sync state is saved in `data/sync_state.json`.

### Large Mailbox Handling

For large mailboxes (10,000+ emails):

1. Use incremental sync
2. Limit emails per folder: `--max 1000`
3. Start with important folders: `--folder INBOX --folder Sent`
4. Use multiple smaller syncs rather than one large sync

### Error Recovery

The scraper handles common errors:

- Network timeouts: Automatically retries
- Authentication errors: Prompts to re-authenticate
- Parse errors: Skips problematic emails and continues

Errors are logged and shown in the final statistics.

## Email Data Structure

Each scraped email contains:

```json
{
  "id": "unique_message_id",
  "from_address": "sender@example.com",
  "to_address": "you@gmail.com",
  "cc_address": "cc@example.com",
  "subject": "Email subject",
  "body": "Email body text",
  "timestamp": "2025-11-10T14:30:22+00:00",
  "category": "inbox",
  "priority": "medium",
  "attachments": ["file1.pdf", "image.jpg"],
  "is_read": false,
  "folder": "INBOX"
}
```

## Security Notes

### Token Security

- OAuth tokens are stored in `config/gmail_token.pickle`
- Keep this file secure and don't commit it to version control
- Tokens can be revoked from your Google Account settings

### Permissions

The scraper requests these Gmail permissions:

- `gmail.readonly`: Read emails and metadata
- `gmail.modify`: Mark emails as read (optional)

### Data Privacy

- All data stays on your local machine
- No emails are sent to external services
- You control what data is scraped and stored

## Troubleshooting

### Common Issues

**"Client secret file not found"**

- Make sure `config/client_secret.json` exists
- Download from Google Cloud Console

**"Authentication failed"**

- Delete `config/gmail_token.pickle` and re-authenticate
- Check if 2FA is enabled (may require app-specific password)

**"Folder not found"**

- Use exact folder names (case-sensitive)
- Run without --folder to see available folders

**"Rate limit exceeded"**

- Gmail has limits on API requests
- Use smaller batch sizes with --max parameter
- Wait a few minutes and retry

### Getting Help

1. Check the logs in `logs/` directory
2. Run with verbose mode for detailed output
3. Check sync state in `data/sync_state.json`

## Performance Tips

### Optimization

- Use incremental sync for regular updates
- Limit email count for faster processing: `--max 500`
- Process important folders first
- Run during off-peak hours for better performance

### Monitoring

The scraper provides detailed statistics:

- Emails processed per second
- Total time taken
- Errors encountered
- Folders processed

## Integration with MailMate

### Dashboard Usage

Once emails are scraped:

1. Start backend: `python mailmate_server.py`
2. Start frontend: `cd frontend && npm start`
3. Visit `http://localhost:3000`
4. Your real emails will appear in all components

### API Integration

Real emails work with all MailMate features:

- Email classification
- Summarization
- Text-to-speech
- Analytics
- Search and filtering

## Example Workflows

### Daily Sync

Create a daily sync routine:

```powershell
# Quick daily sync
python backend/cli.py sync your.email@gmail.com --folder INBOX --limit 50
```

### Weekly Backup

Full backup of important folders:

```powershell
python backend/cli.py scrape your.email@gmail.com \
  --folder INBOX \
  --folder "[Gmail]/Sent Mail" \
  --folder "Important" \
  --output "weekly_backup.json"
```

### Project Migration

Moving from synthetic to real data:

1. Run initial full sync: `python scrape_emails.py`
2. Choose "Standard sync" option
3. Restart MailMate server
4. Verify real emails appear in dashboard
5. Set up regular incremental syncs

This completes the transition from synthetic to real email data in your MailMate system!
