#!/usr/bin/env python3
"""
Basic Structure Testing for BrainWave Analyzer
Tests project structure without requiring dependencies
"""

import os
import sys
from pathlib import Path

def test_project_structure():
    """Test if all required files and directories exist"""
    print("🔍 Testing project structure...")
    
    # Required directories
    required_dirs = [
        'data', 'models', 'training', 'evaluation', 'utils', 'checkpoints'
    ]
    
    # Required files
    required_files = [
        'api.py', 'app.py', 'models.py', 'run_demo.py', 
        'requirements.txt', 'README.md', 'test_setup.py'
    ]
    
    # Required module files
    required_modules = [
        'data/__init__.py', 'data/synthetic_generator.py', 'data/image_generator.py', 'data/dataset.py',
        'models/__init__.py', 'models/eeg_to_image.py', 'models/image_to_eeg.py', 'models/attention.py', 'models/losses.py',
        'training/__init__.py', 'training/train.py', 'training/config.py',
        'evaluation/__init__.py', 'evaluation/metrics.py',
        'utils/__init__.py', 'utils/signal_processing.py', 'utils/visualization.py'
    ]
    
    all_passed = True
    
    # Test directories
    print("\n📁 Testing directories:")
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - Missing")
            all_passed = False
    
    # Test main files
    print("\n📄 Testing main files:")
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing")
            all_passed = False
    
    # Test module files
    print("\n📦 Testing module files:")
    for module in required_modules:
        if Path(module).exists():
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} - Missing")
            all_passed = False
    
    return all_passed

def test_file_content():
    """Test if key files have required content"""
    print("\n📝 Testing file content...")
    
    all_passed = True
    
    # Test API file
    if Path('api.py').exists():
        with open('api.py', 'r') as f:
            api_content = f.read()
        
        required_api_content = ['FastAPI', 'eeg_to_image', 'image_to_eeg', 'models']
        for content in required_api_content:
            if content in api_content:
                print(f"  ✅ api.py contains {content}")
            else:
                print(f"  ❌ api.py missing {content}")
                all_passed = False
    
    # Test Streamlit app
    if Path('app.py').exists():
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        required_app_content = ['streamlit', 'EEG → Image', 'Image → EEG', 'Analysis']
        for content in required_app_content:
            if content in app_content:
                print(f"  ✅ app.py contains {content}")
            else:
                print(f"  ❌ app.py missing {content}")
                all_passed = False
    
    # Test demo launcher
    if Path('run_demo.py').exists():
        with open('run_demo.py', 'r') as f:
            launcher_content = f.read()
        
        required_launcher_content = ['BrainWaveDemoLauncher', 'check_dependencies', 'start_api_server']
        for content in required_launcher_content:
            if content in launcher_content:
                print(f"  ✅ run_demo.py contains {content}")
            else:
                print(f"  ❌ run_demo.py missing {content}")
                all_passed = False
    
    return all_passed

def test_python_version():
    """Test Python version compatibility"""
    print("\n🐍 Testing Python version...")
    
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print("  ✅ Python version compatible (3.8+)")
        return True
    else:
        print("  ❌ Python version too old (need 3.8+)")
        return False

def test_working_directory():
    """Test if we're in the correct directory"""
    print("\n📂 Testing working directory...")
    
    current_dir = os.getcwd()
    print(f"  Current directory: {current_dir}")
    
    # Check if we're in the right place
    if Path('api.py').exists() and Path('app.py').exists():
        print("  ✅ In correct directory (brainwave_app/)")
        return True
    else:
        print("  ❌ Not in correct directory")
        print("  Please run from: BrainWave/backend/brainwave_app/")
        return False

def main():
    """Main testing function"""
    print("🧠 BrainWave Analyzer - Basic Structure Test")
    print("=" * 50)
    
    # Run tests
    structure_ok = test_project_structure()
    content_ok = test_file_content()
    python_ok = test_python_version()
    directory_ok = test_working_directory()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    all_passed = structure_ok and content_ok and python_ok and directory_ok
    
    if all_passed:
        print("✅ ALL BASIC TESTS PASSED!")
        print("\n🚀 Project structure is ready!")
        print("\n📋 Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run demo: python run_demo.py")
        print("  3. Or test setup: python test_setup.py")
    else:
        print("❌ Some tests failed!")
        print("\n🔧 Please check the issues above and fix them.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
