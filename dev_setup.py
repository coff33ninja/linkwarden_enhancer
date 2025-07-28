#!/usr/bin/env python3
"""
Development setup script for Linkwarden Enhancer
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor != 11:
        print(f"❌ Python 3.11 required, found {version.major}.{version.minor}.{version.micro}")
        print("   If you have multiple Python versions installed, try:")
        if platform.system() == "Windows":
            print("   • py -3.11 setup.py")
            print("   • python3.11 setup.py")
        else:
            print("   • python3.11 setup.py")
            print("   • /usr/bin/python3.11 setup.py")
        print("   Or install Python 3.11 if not available")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def find_python311():
    """Find the correct Python 3.11 command"""
    commands_to_try = []
    
    if platform.system() == "Windows":
        commands_to_try = ["py -3.11", "python3.11", "python"]
    else:
        commands_to_try = ["python3.11", "python3", "python"]
    
    for cmd in commands_to_try:
        try:
            result = subprocess.run(f"{cmd} --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0 and "3.11" in result.stdout:
                return cmd
        except:
            continue
    
    return "python"  # fallback


def setup_virtual_environment():
    """Set up virtual environment"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    python_cmd = find_python311()
    return run_command(f"{python_cmd} -m venv .venv", "Creating virtual environment")


def get_activation_command():
    """Get the appropriate activation command for the platform"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"


def install_dependencies():
    """Install dependencies"""
    activation_cmd = get_activation_command()
    
    if platform.system() == "Windows":
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        pip_cmd = ".venv/bin/pip"
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("⚠️  requirements.txt not found, skipping dependency installation")
        return True
    
    commands = [
        (f"{pip_cmd} install --upgrade pip", "Upgrading pip"),
        (f"{pip_cmd} install -r requirements.txt", "Installing dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def create_env_file():
    """Create .env file from example"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("📄 .env file already exists")
        return True
    
    if env_example.exists():
        try:
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from .env.example")
            print("📝 Please edit .env file with your GitHub token and preferences")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("⚠️  .env.example not found, skipping .env creation")
        return True


def create_directories():
    """Create necessary directories"""
    directories = ["data", "backups", "cache", "models", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print(f"✅ Created directories: {', '.join(directories)}")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    if platform.system() == "Windows":
        python_cmd = ".venv\\Scripts\\python"
    else:
        python_cmd = ".venv/bin/python"
    
    # First check if NLTK is installed
    check_command = f'{python_cmd} -c "import nltk; print(\'NLTK is available\')"'
    print("🔍 Checking if NLTK is installed...")
    
    try:
        result = subprocess.run(check_command, shell=True, check=True, capture_output=True, text=True)
        print("✅ NLTK is installed")
    except subprocess.CalledProcessError:
        print("⚠️  NLTK not found, skipping NLTK data download")
        print("   This step will be completed after dependencies are properly installed")
        return True
    
    nltk_command = f'{python_cmd} -c "import nltk; nltk.download(\'vader_lexicon\'); nltk.download(\'punkt\'); nltk.download(\'stopwords\')"'
    
    return run_command(nltk_command, "Downloading NLTK data")


def main():
    """Main setup function"""
    print("🚀 Setting up Linkwarden Enhancer development environment")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Environment File", create_env_file),
        ("Directories", create_directories),
        ("NLTK Data", download_nltk_data),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_function():
            failed_steps.append(step_name)
    
    print("\n" + "=" * 60)
    
    if failed_steps:
        print(f"❌ Setup completed with errors in: {', '.join(failed_steps)}")
        print("\nPlease resolve the errors and run the script again.")
        sys.exit(1)
    else:
        print("✅ Development environment setup completed successfully!")
        
        activation_cmd = get_activation_command()
        print(f"\n🎯 Next steps:")
        print(f"   1. Activate virtual environment: {activation_cmd}")
        print(f"   2. Edit .env file with your GitHub token")
        print(f"   3. Test the application: python cli.py --help")
        print(f"   4. Run with your backup: python cli.py process backup.json output.json --dry-run")
        
        print(f"\n📚 Quick commands:")
        print(f"   • Help: python cli.py help")
        print(f"   • Interactive menu: python cli.py menu")
        print(f"   • Import GitHub: python cli.py import --github --github-token TOKEN -o output.json")
        print(f"   • Process with AI: python cli.py process input.json output.json --enable-ai-analysis")


if __name__ == "__main__":
    main()