# Gmail Connectivity Troubleshooting Guide

## Problem: WinError 10060 (Connection Timeout)

This error indicates that your computer cannot reach Gmail's servers. Here are solutions:

### 1. **Corporate Network/Firewall Issues**

If you're on a corporate network:

```powershell
# Check if ports are blocked
Test-NetConnection smtp.gmail.com -Port 993
Test-NetConnection imap.gmail.com -Port 993
Test-NetConnection gmail.googleapis.com -Port 443
```

**Solutions:**

- Try from home network or personal hotspot
- Ask IT to whitelist Gmail API domains
- Use VPN to bypass corporate restrictions

### 2. **Windows Firewall**

```powershell
# Temporarily disable Windows Firewall (as Administrator)
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Test Gmail scraping, then re-enable
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
```

### 3. **Proxy Configuration**

If using corporate proxy:

```powershell
# Set proxy for Python (if needed)
set HTTP_PROXY=http://proxy.company.com:8080
set HTTPS_PROXY=http://proxy.company.com:8080

# Or disable proxy temporarily
set HTTP_PROXY=
set HTTPS_PROXY=
```

### 4. **DNS Issues**

```powershell
# Flush DNS cache
ipconfig /flushdns

# Try different DNS servers (Google DNS)
netsh interface ip set dns "Wi-Fi" static 8.8.8.8
netsh interface ip add dns "Wi-Fi" 8.8.4.4 index=2
```

### 5. **Alternative Network Test**

```powershell
# Test basic connectivity
ping gmail.com
nslookup gmail.googleapis.com

# Test HTTPS connectivity
curl -I https://gmail.googleapis.com
```

## Temporary Workarounds

### Option A: Use Different Network

- Switch to mobile hotspot
- Try from home network
- Use different WiFi

### Option B: Gmail Takeout (Manual Export)

1. Go to [Google Takeout](https://takeout.google.com)
2. Select "Gmail"
3. Choose "JSON" format
4. Download your email archive
5. Use our import tool (see below)

### Option C: Email Import Tool

Create a manual import script for downloaded Gmail data:

```python
# Will create this if you need the manual import option
```

## Testing Commands

```powershell
# Test Gmail API connectivity
python -c "import requests; print(requests.get('https://gmail.googleapis.com').status_code)"

# Test OAuth flow
python -c "from google.auth.transport.requests import Request; print('OAuth imports working')"
```

## Next Steps Based on Your Network

1. **Corporate Network**: Talk to IT about Gmail API access
2. **Home Network**: Check router/ISP restrictions
3. **VPN**: Try connecting without VPN
4. **Antivirus**: Check if antivirus is blocking connections

## Success Verification

Once network issues are resolved, you should see:

```
✅ Connected successfully!
📥 Fetching emails from INBOX...
💾 Saving X emails to data/gmail_emails_YYYYMMDD_HHMMSS.json
🎉 Your emails are now saved and ready to use!
```
