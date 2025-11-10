"""
MailMate CLI Interface

Command-line interface for MailMate Dashboard with support for:
- Demo mode with sample data generation
- Email processing operations
- Dashboard server management
- Configuration management
"""

import typer
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv, set_key, find_dotenv
import threading
import signal
import webbrowser
from datetime import datetime

# Add backend directory to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from logging_config import setup_logging, get_logger, log_startup, log_shutdown
from email_loader import EmailDataLoader
from email_classifier import EmailClassifier  
from email_summarizer import EmailSummarizer
from text_to_speech import TextToSpeech
from email_scraper import EmailScraper

# Initialize CLI app
app = typer.Typer(
    name="mailmate",
    help="MailMate Dashboard - AI-powered email management system",
    add_completion=False
)

# Global variables for server processes
_server_processes: Dict[str, subprocess.Popen] = {}
_shutdown_event = threading.Event()


class ConfigManager:
    """Manages configuration files and environment variables."""
    
    def __init__(self, config_file: str = ".env"):
        self.config_file = Path(config_file)
        self.env_file = find_dotenv() or self.config_file
        load_dotenv(self.env_file)
        
        # Default configuration
        self.defaults = {
            "MAILMATE_LOG_LEVEL": "INFO",
            "MAILMATE_LOG_DIR": "logs",
            "MAILMATE_DATA_DIR": "data",
            "MAILMATE_API_HOST": "0.0.0.0",
            "MAILMATE_API_PORT": "5000",
            "MAILMATE_FRONTEND_PORT": "3000",
            "MAILMATE_DEBUG": "False",
            "MAILMATE_AUTO_OPEN_BROWSER": "True",
            "MAILMATE_DEMO_EMAIL_COUNT": "50",
            "MAILMATE_TTS_ENGINE": "gTTS",
            "MAILMATE_TTS_LANGUAGE": "en",
            "MAILMATE_AUDIO_OUTPUT_DIR": "audio_outputs"
        }
        
        self._ensure_config_exists()
    
    def _ensure_config_exists(self):
        """Ensure configuration file exists with default values."""
        if not self.config_file.exists():
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            f.write("# MailMate Configuration\n")
            f.write("# Generated automatically - modify as needed\n\n")
            
            for key, value in self.defaults.items():
                f.write(f"{key}={value}\n")
        
        typer.echo(f"✅ Created default configuration: {self.config_file}")
    
    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get configuration value."""
        return os.getenv(key, default or self.defaults.get(key, ""))
    
    def set(self, key: str, value: str):
        """Set configuration value."""
        set_key(self.env_file, key, value)
        os.environ[key] = value
    
    def get_all(self) -> Dict[str, str]:
        """Get all configuration values."""
        config = {}
        for key in self.defaults.keys():
            config[key] = self.get(key)
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check required directories
        for dir_key in ["MAILMATE_LOG_DIR", "MAILMATE_DATA_DIR", "MAILMATE_AUDIO_OUTPUT_DIR"]:
            dir_path = Path(self.get(dir_key))
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")
        
        # Check ports
        try:
            api_port = int(self.get("MAILMATE_API_PORT"))
            if not (1024 <= api_port <= 65535):
                issues.append("API port must be between 1024 and 65535")
        except ValueError:
            issues.append("API port must be a valid integer")
        
        try:
            frontend_port = int(self.get("MAILMATE_FRONTEND_PORT"))
            if not (1024 <= frontend_port <= 65535):
                issues.append("Frontend port must be between 1024 and 65535")
        except ValueError:
            issues.append("Frontend port must be a valid integer")
        
        return issues


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        typer.echo("\n🛑 Shutdown signal received. Cleaning up...")
        _shutdown_event.set()
        stop_all_servers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def stop_all_servers():
    """Stop all running server processes."""
    for name, process in _server_processes.items():
        if process and process.poll() is None:
            typer.echo(f"🛑 Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    _server_processes.clear()


@app.command("config")
def config_command(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all configuration values"),
    get_key: Optional[str] = typer.Option(None, "--get", "-g", help="Get specific configuration value"),
    set_key: Optional[str] = typer.Option(None, "--set", "-s", help="Set configuration key"),
    set_value: Optional[str] = typer.Option(None, "--value", "-v", help="Set configuration value"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration")
):
    """Manage MailMate configuration."""
    config = ConfigManager()
    
    if reset:
        if typer.confirm("⚠️  Reset configuration to defaults?"):
            config.create_default_config()
            typer.echo("✅ Configuration reset to defaults")
        return
    
    if validate:
        issues = config.validate()
        if issues:
            typer.echo("❌ Configuration issues found:")
            for issue in issues:
                typer.echo(f"  • {issue}")
        else:
            typer.echo("✅ Configuration is valid")
        return
    
    if set_key and set_value:
        config.set(set_key, set_value)
        typer.echo(f"✅ Set {set_key} = {set_value}")
        return
    
    if get_key:
        value = config.get(get_key)
        typer.echo(f"{get_key} = {value}")
        return
    
    if list_all:
        typer.echo("📋 Current configuration:")
        config_data = config.get_all()
        for key, value in config_data.items():
            typer.echo(f"  {key} = {value}")
        return
    
    # Default: show help
    typer.echo("Use --help to see available options")


@app.command("demo")
def demo_command(
    email_count: int = typer.Option(50, "--count", "-c", help="Number of sample emails to generate"),
    categories: Optional[List[str]] = typer.Option(None, "--categories", help="Email categories to generate"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for sample data"),
    format_type: str = typer.Option("json", "--format", "-f", help="Output format (json, csv)")
):
    """Generate sample email data for demonstration."""
    config = ConfigManager()
    
    # Setup logging
    setup_logging({
        "log_level": config.get("MAILMATE_LOG_LEVEL"),
        "log_dir": config.get("MAILMATE_LOG_DIR")
    })
    
    logger = get_logger("mailmate.demo")
    log_startup(logger, "MailMate Demo Mode", "1.0.0")
    
    try:
        # Set default categories if not provided
        if not categories:
            categories = ["Work", "Personal", "Finance", "Shopping", "Travel", "Urgent"]
        
        # Set output directory
        if not output_dir:
            output_dir = config.get("MAILMATE_DATA_DIR")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        typer.echo(f"🚀 Generating {email_count} sample emails...")
        typer.echo(f"📁 Output directory: {output_path}")
        typer.echo(f"📋 Categories: {', '.join(categories)}")
        
        # Generate sample emails
        start_time = time.time()
        
        # Create email loader and generate samples
        loader = EmailDataLoader()
        
        with typer.progressbar(length=email_count, label="Generating emails") as progress:
            emails = []
            for i in range(email_count):
                # Simulate email generation (in real implementation, use EmailDataLoader methods)
                email = {
                    "id": f"demo_{i+1:03d}",
                    "subject": f"Demo Email {i+1}",
                    "from_address": f"sender{i+1}@example.com",
                    "to_address": "user@mailmate.com",
                    "body": f"This is demo email content for email {i+1}.",
                    "category": categories[i % len(categories)],
                    "priority": ["High", "Medium", "Low"][i % 3],
                    "timestamp": datetime.now().isoformat(),
                    "is_read": i % 3 == 0,
                    "attachments": []
                }
                emails.append(email)
                progress.update(1)
                time.sleep(0.01)  # Simulate processing time
        
        # Save to file
        if format_type.lower() == "json":
            output_file = output_path / "demo_emails.json"
            with open(output_file, 'w') as f:
                json.dump(emails, f, indent=2)
        elif format_type.lower() == "csv":
            import csv
            output_file = output_path / "demo_emails.csv"
            with open(output_file, 'w', newline='') as f:
                if emails:
                    writer = csv.DictWriter(f, fieldnames=emails[0].keys())
                    writer.writeheader()
                    writer.writerows(emails)
        
        duration = time.time() - start_time
        
        typer.echo(f"✅ Generated {len(emails)} emails in {duration:.2f}s")
        typer.echo(f"📄 Saved to: {output_file}")
        
        logger.info(f"Demo data generation completed: {len(emails)} emails in {duration:.2f}s")
        
    except Exception as e:
        logger.error(f"Demo generation failed: {e}", exc_info=True)
        typer.echo(f"❌ Error generating demo data: {e}")
        raise typer.Exit(1)


@app.command("process")
def process_command(
    input_file: str = typer.Argument(..., help="Input email file to process"),
    operation: str = typer.Option("all", "--operation", "-op", help="Operation to perform (classify, summarize, tts, all)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    batch_size: int = typer.Option(10, "--batch", "-b", help="Batch size for processing")
):
    """Process emails with AI operations (classification, summarization, TTS)."""
    config = ConfigManager()
    
    # Setup logging
    setup_logging({
        "log_level": config.get("MAILMATE_LOG_LEVEL"),
        "log_dir": config.get("MAILMATE_LOG_DIR")
    })
    
    logger = get_logger("mailmate.process")
    log_startup(logger, "MailMate Email Processor", "1.0.0")
    
    try:
        input_path = Path(input_file)
        if not input_path.exists():
            typer.echo(f"❌ Input file not found: {input_path}")
            raise typer.Exit(1)
        
        # Set output directory
        if not output_dir:
            output_dir = config.get("MAILMATE_DATA_DIR")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        typer.echo(f"📧 Processing emails from: {input_path}")
        typer.echo(f"🔄 Operation: {operation}")
        typer.echo(f"📁 Output directory: {output_path}")
        
        # Load emails
        with open(input_path, 'r') as f:
            emails = json.load(f)
        
        typer.echo(f"📊 Found {len(emails)} emails to process")
        
        results = []
        
        with typer.progressbar(emails, label="Processing emails") as progress:
            for email in progress:
                try:
                    result = {"email_id": email.get("id", "unknown")}
                    
                    if operation in ["classify", "all"]:
                        # Email classification
                        classifier = EmailClassifier()
                        classification = classifier.classify_email(email.get("body", ""))
                        result["classification"] = classification
                        logger.info(f"Classified email {email['id']}: {classification}")
                    
                    if operation in ["summarize", "all"]:
                        # Email summarization
                        summarizer = EmailSummarizer()
                        summary = summarizer.summarize_email(email.get("body", ""))
                        result["summary"] = summary
                        logger.info(f"Summarized email {email['id']}")
                    
                    if operation in ["tts", "all"]:
                        # Text-to-speech
                        tts = TextToSpeech(
                            engine=config.get("MAILMATE_TTS_ENGINE"),
                            language=config.get("MAILMATE_TTS_LANGUAGE"),
                            output_dir=config.get("MAILMATE_AUDIO_OUTPUT_DIR")
                        )
                        audio_file = tts.text_to_speech(
                            email.get("body", ""),
                            filename=f"email_{email.get('id', 'unknown')}"
                        )
                        result["audio_file"] = str(audio_file) if audio_file else None
                        logger.info(f"Generated TTS for email {email['id']}")
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing email {email.get('id', 'unknown')}: {e}")
                    results.append({
                        "email_id": email.get("id", "unknown"),
                        "error": str(e)
                    })
        
        # Save results
        results_file = output_path / f"processing_results_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        typer.echo(f"✅ Processing completed")
        typer.echo(f"📄 Results saved to: {results_file}")
        
        # Print summary
        successful = len([r for r in results if "error" not in r])
        failed = len(results) - successful
        typer.echo(f"📊 Summary: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"Email processing failed: {e}", exc_info=True)
        typer.echo(f"❌ Error processing emails: {e}")
        raise typer.Exit(1)


@app.command("server")
def server_command(
    component: str = typer.Option("all", "--component", "-c", help="Component to start (api, frontend, all)"),
    host: Optional[str] = typer.Option(None, "--host", help="API host"),
    api_port: Optional[int] = typer.Option(None, "--api-port", help="API port"),
    frontend_port: Optional[int] = typer.Option(None, "--frontend-port", help="Frontend port"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically")
):
    """Start MailMate server components."""
    config = ConfigManager()
    
    # Setup logging
    setup_logging({
        "log_level": config.get("MAILMATE_LOG_LEVEL"),
        "log_dir": config.get("MAILMATE_LOG_DIR")
    })
    
    logger = get_logger("mailmate.server")
    log_startup(logger, "MailMate Server", "1.0.0")
    
    setup_signal_handlers()
    
    try:
        # Use config values or provided values
        api_host = host or config.get("MAILMATE_API_HOST")
        api_port_num = api_port or int(config.get("MAILMATE_API_PORT"))
        frontend_port_num = frontend_port or int(config.get("MAILMATE_FRONTEND_PORT"))
        
        # Validate configuration
        issues = config.validate()
        if issues:
            for issue in issues:
                typer.echo(f"⚠️  {issue}")
        
        # Start API server
        if component in ["api", "all"]:
            typer.echo(f"🚀 Starting API server on {api_host}:{api_port_num}")
            
            api_cmd = [
                sys.executable, 
                str(backend_dir / "complete_integration_with_audio.py"),
                "--host", api_host,
                "--port", str(api_port_num)
            ]
            
            if debug:
                api_cmd.append("--debug")
            
            api_process = subprocess.Popen(
                api_cmd,
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            _server_processes["API"] = api_process
            logger.info(f"API server started with PID {api_process.pid}")
            
            # Wait a moment for API to start
            time.sleep(3)
        
        # Start frontend server
        if component in ["frontend", "all"]:
            frontend_dir = backend_dir.parent / "frontend"
            
            if not frontend_dir.exists():
                typer.echo(f"❌ Frontend directory not found: {frontend_dir}")
                raise typer.Exit(1)
            
            typer.echo(f"🌐 Starting frontend server on port {frontend_port_num}")
            
            # Set frontend port in environment
            env = os.environ.copy()
            env["PORT"] = str(frontend_port_num)
            
            frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            _server_processes["Frontend"] = frontend_process
            logger.info(f"Frontend server started with PID {frontend_process.pid}")
            
            # Wait for frontend to compile
            time.sleep(10)
        
        # Open browser
        if not no_browser and config.get("MAILMATE_AUTO_OPEN_BROWSER").lower() == "true":
            if component in ["frontend", "all"]:
                browser_url = f"http://localhost:{frontend_port_num}"
                typer.echo(f"🌐 Opening browser: {browser_url}")
                webbrowser.open(browser_url)
        
        # Display running services
        typer.echo("🎉 MailMate is now running!")
        typer.echo("")
        typer.echo("📍 Available services:")
        
        if "API" in _server_processes:
            typer.echo(f"  🔧 API Server: http://{api_host}:{api_port_num}")
            typer.echo(f"  🩺 Health Check: http://{api_host}:{api_port_num}/api/health")
        
        if "Frontend" in _server_processes:
            typer.echo(f"  🌐 Dashboard: http://localhost:{frontend_port_num}")
        
        typer.echo("")
        typer.echo("💡 Press Ctrl+C to stop all servers")
        
        # Monitor processes
        while not _shutdown_event.is_set():
            time.sleep(1)
            
            # Check if any process has died
            for name, process in _server_processes.copy().items():
                if process.poll() is not None:
                    typer.echo(f"⚠️  {name} server has stopped")
                    logger.warning(f"{name} server process ended with code {process.returncode}")
                    del _server_processes[name]
            
            if not _server_processes:
                typer.echo("🛑 All servers have stopped")
                break
    
    except KeyboardInterrupt:
        typer.echo("\n🛑 Received shutdown signal")
    except Exception as e:
        logger.error(f"Server startup failed: {e}", exc_info=True)
        typer.echo(f"❌ Error starting servers: {e}")
        raise typer.Exit(1)
    finally:
        stop_all_servers()
        log_shutdown(logger, "MailMate Server")


@app.command("status")
def status_command():
    """Show MailMate system status."""
    config = ConfigManager()
    
    typer.echo("📊 MailMate System Status")
    typer.echo("=" * 40)
    
    # Configuration status
    typer.echo("\n🔧 Configuration:")
    issues = config.validate()
    if issues:
        typer.echo("  ❌ Configuration issues found:")
        for issue in issues:
            typer.echo(f"    • {issue}")
    else:
        typer.echo("  ✅ Configuration is valid")
    
    # Check dependencies
    typer.echo("\n📦 Dependencies:")
    dependencies = [
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS"),
        ("flask_socketio", "Flask-SocketIO"),
        ("dotenv", "python-dotenv"),
        ("typer", "Typer")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            typer.echo(f"  ✅ {name}")
        except ImportError:
            typer.echo(f"  ❌ {name} (not installed)")
    
    # Check directories
    typer.echo("\n📁 Directories:")
    directories = [
        ("MAILMATE_LOG_DIR", "Logs"),
        ("MAILMATE_DATA_DIR", "Data"),
        ("MAILMATE_AUDIO_OUTPUT_DIR", "Audio Output")
    ]
    
    for config_key, name in directories:
        dir_path = Path(config.get(config_key))
        if dir_path.exists():
            typer.echo(f"  ✅ {name}: {dir_path}")
        else:
            typer.echo(f"  ⚠️  {name}: {dir_path} (not found)")
    
    # Check running processes
    typer.echo("\n🔄 Running Processes:")
    if _server_processes:
        for name, process in _server_processes.items():
            if process.poll() is None:
                typer.echo(f"  ✅ {name} (PID: {process.pid})")
            else:
                typer.echo(f"  ❌ {name} (stopped)")
    else:
        typer.echo("  ℹ️  No MailMate processes running")


@app.command("logs")
def logs_command(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    module: Optional[str] = typer.Option(None, "--module", "-m", help="Specific module log to show")
):
    """View MailMate logs."""
    config = ConfigManager()
    log_dir = Path(config.get("MAILMATE_LOG_DIR"))
    
    if not log_dir.exists():
        typer.echo(f"❌ Log directory not found: {log_dir}")
        raise typer.Exit(1)
    
    # Select log file
    if module:
        log_file = log_dir / f"{module}.log"
        if not log_file.exists():
            typer.echo(f"❌ Module log not found: {log_file}")
            raise typer.Exit(1)
    else:
        log_file = log_dir / "mailmate.log"
        if not log_file.exists():
            typer.echo(f"❌ Main log not found: {log_file}")
            raise typer.Exit(1)
    
    typer.echo(f"📄 Viewing: {log_file}")
    typer.echo("-" * 60)
    
    if follow:
        # Follow mode - tail -f equivalent
        import subprocess
        try:
            if sys.platform == "win32":
                # Windows doesn't have tail, use PowerShell
                subprocess.run([
                    "powershell", "-Command", 
                    f"Get-Content '{log_file}' -Tail {lines} -Wait"
                ])
            else:
                subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
        except KeyboardInterrupt:
            typer.echo("\n📄 Log following stopped")
    else:
        # Show last N lines
        try:
            with open(log_file, 'r') as f:
                lines_list = f.readlines()
                for line in lines_list[-lines:]:
                    typer.echo(line.rstrip())
        except Exception as e:
            typer.echo(f"❌ Error reading log file: {e}")


@app.command("scrape")
def scrape_emails(
    email: str = typer.Argument(..., help="Gmail address to scrape"),
    folders: Optional[List[str]] = typer.Option(None, "--folder", "-f", help="Specific folders to scrape"),
    max_emails: Optional[int] = typer.Option(None, "--max", "-m", help="Maximum emails per folder"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename"),
    format: str = typer.Option("json", "--format", help="Output format (json/csv)"),
    incremental: bool = typer.Option(True, "--incremental/--full", help="Incremental sync (default) vs full sync"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory for storing emails")
):
    """
    Scrape emails from Gmail account and save to file.
    
    This command connects to your Gmail account using OAuth2 and downloads
    all emails from specified folders. Use incremental sync to only fetch
    new emails since last sync.
    
    Examples:
        mailmate scrape your.email@gmail.com
        mailmate scrape your.email@gmail.com --folder INBOX --folder Sent
        mailmate scrape your.email@gmail.com --max 1000 --format csv
        mailmate scrape your.email@gmail.com --full --output my_emails.json
    """
    logger = get_logger("scrape")
    
    typer.echo("🚀 Starting Gmail email scraping...")
    typer.echo(f"📧 Email: {email}")
    typer.echo(f"📂 Data directory: {data_dir}")
    
    if folders:
        typer.echo(f"📁 Folders: {', '.join(folders)}")
    else:
        typer.echo("📁 Folders: All available folders")
    
    if max_emails:
        typer.echo(f"📊 Max emails per folder: {max_emails}")
    
    typer.echo(f"🔄 Sync type: {'Incremental' if incremental else 'Full'}")
    typer.echo(f"📄 Output format: {format}")
    
    # Create data directory
    Path(data_dir).mkdir(exist_ok=True)
    
    try:
        with EmailScraper(data_dir=data_dir) as scraper:
            # Connect to Gmail
            typer.echo("\n🔐 Connecting to Gmail...")
            scraper.connect_gmail(email)
            
            # Show available folders if none specified
            if not folders:
                available_folders = scraper.get_folder_list()
                typer.echo(f"📋 Available folders: {', '.join(available_folders)}")
            
            # Scrape emails
            typer.echo("\n📥 Scraping emails...")
            emails = scraper.scrape_all_folders(
                folders=folders,
                max_emails_per_folder=max_emails,
                incremental=incremental
            )
            
            if not emails:
                typer.echo("⚠️  No emails found to scrape")
                return
            
            # Save emails
            typer.echo(f"\n💾 Saving {len(emails)} emails...")
            file_path = scraper.save_emails(
                emails=emails,
                filename=output,
                format=format
            )
            
            # Show statistics
            stats = scraper.get_stats()
            typer.echo("\n✅ Scraping completed!")
            typer.echo(f"📊 Statistics:")
            typer.echo(f"   • Total emails scraped: {stats['total_new']}")
            typer.echo(f"   • Total emails processed: {stats['total_processed']}")
            typer.echo(f"   • Folders scraped: {stats['total_folders']}")
            typer.echo(f"   • Output file: {file_path}")
            
            if stats.get('duration_seconds'):
                typer.echo(f"   • Duration: {stats['duration_seconds']:.1f} seconds")
                typer.echo(f"   • Speed: {stats.get('emails_per_second', 0):.1f} emails/second")
            
            if stats['errors']:
                typer.echo(f"   • Errors: {len(stats['errors'])}")
                for error in stats['errors'][:5]:  # Show first 5 errors
                    typer.echo(f"     - {error}")
                if len(stats['errors']) > 5:
                    typer.echo(f"     ... and {len(stats['errors']) - 5} more errors")
            
            typer.echo(f"\n🎉 Emails successfully saved to: {file_path}")
            
    except Exception as e:
        logger.error(f"Email scraping failed: {e}")
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command("sync")
def sync_emails(
    email: str = typer.Argument(..., help="Gmail address to sync"),
    folder: str = typer.Option("INBOX", "--folder", "-f", help="Folder to sync"),
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum emails to fetch"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory")
):
    """
    Quick sync of specific folder using API endpoints.
    
    This is a simpler alternative to the full scrape command that uses
    the backend API endpoints to sync a specific folder.
    """
    logger = get_logger("sync")
    
    typer.echo(f"🔄 Syncing {folder} for {email}...")
    
    try:
        from email_scraper import EmailScraper
        
        with EmailScraper(data_dir=data_dir) as scraper:
            scraper.connect_gmail(email)
            
            emails = scraper.scrape_folder(
                folder_name=folder,
                max_emails=limit,
                incremental=True
            )
            
            if emails:
                filename = output or f"sync_{folder.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                file_path = scraper.save_emails(emails, filename=filename)
                
                typer.echo(f"✅ Synced {len(emails)} emails from {folder}")
                typer.echo(f"📁 Saved to: {file_path}")
            else:
                typer.echo(f"ℹ️  No new emails in {folder}")
                
    except Exception as e:
        logger.error(f"Email sync failed: {e}")
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", help="Show version information"),
    config_file: str = typer.Option(".env", "--config", help="Configuration file path")
):
    """
    MailMate Dashboard - AI-powered email management system
    
    A comprehensive email management platform with AI classification,
    summarization, text-to-speech, and real-time analytics.
    """
    if version:
        typer.echo("MailMate Dashboard v1.0.0")
        typer.echo("AI-powered email management system")
        raise typer.Exit()


if __name__ == "__main__":
    app()