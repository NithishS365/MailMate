#!/usr/bin/env python3
"""
MailMate Setup Script

Automated setup script for MailMate Dashboard that:
- Creates virtual environment (optional)
- Installs Python dependencies
- Installs Node.js dependencies for frontend
- Sets up default configuration
- Validates installation
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
import platform

def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"🔄 Running: {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        if isinstance(command, str):
            # For PowerShell commands on Windows
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command", command],
                    cwd=cwd,
                    check=check,
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    check=check,
                    capture_output=True,
                    text=True
                )
        else:
            result = subprocess.run(
                command,
                cwd=cwd,
                check=check,
                capture_output=True,
                text=True
            )
        
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"📤 stdout: {e.stdout}")
        if e.stderr:
            print(f"📤 stderr: {e.stderr}")
        if check:
            raise
        return e

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_node_version():
    """Check if Node.js is installed and compatible."""
    try:
        result = run_command(["node", "--version"], check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"📦 Node.js version: {version}")
            
            # Extract major version number
            major_version = int(version.lstrip('v').split('.')[0])
            if major_version >= 16:
                print("✅ Node.js version is compatible")
                return True
            else:
                print("⚠️  Node.js 16+ is recommended")
                return True
        else:
            print("❌ Node.js not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Node.js: {e}")
        return False

def check_npm():
    """Check if npm is available."""
    try:
        result = run_command(["npm", "--version"], check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"📦 npm version: {version}")
            print("✅ npm is available")
            return True
        else:
            print("❌ npm not found")
            return False
    except Exception as e:
        print(f"❌ Error checking npm: {e}")
        return False

def create_virtual_environment(venv_path):
    """Create a Python virtual environment."""
    print(f"🔧 Creating virtual environment: {venv_path}")
    
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created")
        return True
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def get_venv_python(venv_path):
    """Get the Python executable path for the virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def get_venv_pip(venv_path):
    """Get the pip executable path for the virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def install_python_dependencies(python_exe, requirements_file):
    """Install Python dependencies."""
    print(f"📦 Installing Python dependencies from {requirements_file}")
    
    try:
        # Upgrade pip first
        run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        run_command([str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)])
        
        print("✅ Python dependencies installed")
        return True
    except Exception as e:
        print(f"❌ Error installing Python dependencies: {e}")
        return False

def install_frontend_dependencies(frontend_dir):
    """Install Node.js dependencies for frontend."""
    print(f"🌐 Installing frontend dependencies in {frontend_dir}")
    
    try:
        run_command(["npm", "install"], cwd=frontend_dir)
        print("✅ Frontend dependencies installed")
        return True
    except Exception as e:
        print(f"❌ Error installing frontend dependencies: {e}")
        return False

def setup_configuration(project_dir):
    """Set up default configuration."""
    print("⚙️  Setting up configuration")
    
    config_file = project_dir / ".env"
    
    if config_file.exists():
        print(f"ℹ️  Configuration file already exists: {config_file}")
        return True
    
    try:
        # Create default .env file
        default_config = """# MailMate Configuration
# Generated automatically - modify as needed

MAILMATE_LOG_LEVEL=INFO
MAILMATE_LOG_DIR=logs
MAILMATE_DATA_DIR=data
MAILMATE_API_HOST=0.0.0.0
MAILMATE_API_PORT=5000
MAILMATE_FRONTEND_PORT=3000
MAILMATE_DEBUG=False
MAILMATE_AUTO_OPEN_BROWSER=True
MAILMATE_DEMO_EMAIL_COUNT=50
MAILMATE_TTS_ENGINE=gTTS
MAILMATE_TTS_LANGUAGE=en
MAILMATE_AUDIO_OUTPUT_DIR=audio_outputs
"""
        
        with open(config_file, 'w') as f:
            f.write(default_config)
        
        print(f"✅ Configuration created: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False

def create_directories(project_dir):
    """Create necessary directories."""
    print("📁 Creating directories")
    
    directories = [
        "logs",
        "data",
        "audio_outputs",
        "audio_outputs/emails",
        "audio_outputs/batch",
        "audio_outputs/tts_features"
    ]
    
    for dir_name in directories:
        dir_path = project_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    return True

def validate_installation(python_exe, project_dir):
    """Validate the installation by running basic imports."""
    print("🔍 Validating installation")
    
    test_script = """
import sys
import importlib

modules = [
    'typer',
    'dotenv',
    'flask',
    'flask_cors',
    'flask_socketio',
    'pandas',
    'sklearn',
    'transformers',
    'gtts',
    'pyttsx3'
]

failed_imports = []

for module in modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module}")
    except ImportError as e:
        print(f"❌ {module}: {e}")
        failed_imports.append(module)

if failed_imports:
    print(f"\\n⚠️  Failed imports: {', '.join(failed_imports)}")
    sys.exit(1)
else:
    print("\\n🎉 All modules imported successfully!")
"""
    
    try:
        result = run_command([str(python_exe), "-c", test_script], cwd=project_dir)
        if result.returncode == 0:
            print("✅ Installation validation successful")
            return True
        else:
            print("❌ Installation validation failed")
            return False
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 MailMate Dashboard Setup")
    print("=" * 50)
    
    # Get project directory
    project_dir = Path(__file__).parent
    backend_dir = project_dir / "backend"
    frontend_dir = project_dir / "frontend"
    
    print(f"📁 Project directory: {project_dir}")
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    has_node = check_node_version()
    has_npm = check_npm() if has_node else False
    
    # Ask user about virtual environment
    use_venv = input("\n❓ Create virtual environment? (y/N): ").lower().startswith('y')
    
    python_exe = sys.executable
    
    if use_venv:
        venv_path = project_dir / "venv"
        if venv_path.exists():
            print(f"ℹ️  Virtual environment already exists: {venv_path}")
        else:
            if not create_virtual_environment(venv_path):
                sys.exit(1)
        
        python_exe = get_venv_python(venv_path)
        print(f"🐍 Using Python: {python_exe}")
    
    # Install Python dependencies
    requirements_file = backend_dir / "requirements.txt"
    if not install_python_dependencies(python_exe, requirements_file):
        print("⚠️  Python dependencies installation failed, but continuing...")
    
    # Install frontend dependencies
    if has_npm and frontend_dir.exists():
        if not install_frontend_dependencies(frontend_dir):
            print("⚠️  Frontend dependencies installation failed, but continuing...")
    else:
        print("⚠️  Skipping frontend setup (Node.js/npm not available or frontend dir missing)")
    
    # Setup configuration and directories
    setup_configuration(project_dir)
    create_directories(project_dir)
    
    # Validate installation
    if not validate_installation(python_exe, project_dir):
        print("⚠️  Installation validation failed, but setup is complete")
    
    # Final instructions
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    
    if use_venv:
        if platform.system() == "Windows":
            print(f"1. Activate virtual environment: .\\venv\\Scripts\\activate")
        else:
            print(f"1. Activate virtual environment: source venv/bin/activate")
    
    print(f"2. Run MailMate: python backend/cli.py --help")
    print(f"3. Start demo: python backend/cli.py demo")
    print(f"4. Start servers: python backend/cli.py server")
    print(f"5. Check status: python backend/cli.py status")
    
    print("\n🔧 Configuration file: .env")
    print("📚 Documentation: README.md")

if __name__ == "__main__":
    main()