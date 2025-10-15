#!/usr/bin/env python3
"""
Demo Verification Script for BrainWave Analyzer
Quick verification that everything is ready for demo
"""

import os
import sys
from pathlib import Path

def check_structure():
    """Check project structure"""
    print("🔍 Checking project structure...")
    
    required_items = [
        # Directories
        'data', 'models', 'training', 'evaluation', 'utils', 'checkpoints',
        # Main files
        'api.py', 'app.py', 'models.py', 'run_demo.py', 'requirements.txt', 'README.md',
        # Key modules
        'data/synthetic_generator.py', 'data/image_generator.py', 'data/dataset.py',
        'models/eeg_to_image.py', 'models/image_to_eeg.py', 'models/attention.py',
        'training/train.py', 'training/config.py',
        'evaluation/metrics.py',
        'utils/signal_processing.py', 'utils/visualization.py'
    ]
    
    missing = []
    for item in required_items:
        if not Path(item).exists():
            missing.append(item)
    
    if missing:
        print(f"❌ Missing items: {missing}")
        return False
    else:
        print("✅ All required files and directories present")
        return True

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_directory():
    """Check working directory"""
    print("📂 Checking working directory...")
    
    if Path('api.py').exists() and Path('app.py').exists():
        print("✅ In correct directory")
        return True
    else:
        print("❌ Not in correct directory")
        print("   Please run from: BrainWave/backend/brainwave_app/")
        return False

def check_file_sizes():
    """Check that key files have content"""
    print("📊 Checking file sizes...")
    
    key_files = ['api.py', 'app.py', 'models.py', 'run_demo.py']
    for file in key_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            if size > 1000:  # At least 1KB
                print(f"✅ {file} ({size:,} bytes)")
            else:
                print(f"⚠️  {file} seems small ({size} bytes)")
        else:
            print(f"❌ {file} missing")

def main():
    """Main verification function"""
    print("🧠 BrainWave Analyzer - Demo Verification")
    print("=" * 50)
    
    # Run checks
    structure_ok = check_structure()
    python_ok = check_python_version()
    directory_ok = check_directory()
    check_file_sizes()
    
    # Summary
    print("\n" + "=" * 50)
    
    if structure_ok and python_ok and directory_ok:
        print("🎉 DEMO VERIFICATION PASSED!")
        print("\n🚀 Your BrainWave Analyzer is ready for demo!")
        print("\n📋 To start the demo:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run: python run_demo.py")
        print("  3. Open browser to the Streamlit interface")
        print("\n🎯 Demo features available:")
        print("  • EEG → Image synthesis")
        print("  • Image → EEG prediction")
        print("  • Interactive visualizations")
        print("  • Real-time signal analysis")
        print("  • Professional API endpoints")
        
        return True
    else:
        print("❌ DEMO VERIFICATION FAILED!")
        print("\n🔧 Please fix the issues above before running the demo.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
