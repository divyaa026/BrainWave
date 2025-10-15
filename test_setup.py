#!/usr/bin/env python3
"""
Test script to verify BrainWave Analyzer setup
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Test if all required directories exist"""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'data', 'models', 'training', 'evaluation', 'utils', 'checkpoints'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - Missing")
            all_exist = False
    
    return all_exist

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'api.py', 'app.py', 'models.py', 'run_demo.py', 'requirements.txt', 'README.md'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing")
            all_exist = False
    
    return all_exist

def test_module_imports():
    """Test if modules can be imported (without dependencies)"""
    print("\n📦 Testing module imports...")
    
    # Test basic Python imports
    basic_modules = ['numpy', 'pandas', 'json', 'time', 'os', 'sys']
    
    for module in basic_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - Not available")
    
    # Test our custom modules (without dependencies)
    try:
        # Test if our modules exist
        if Path('data/__init__.py').exists():
            print("  ✅ data module structure")
        else:
            print("  ❌ data module structure")
        
        if Path('models/__init__.py').exists():
            print("  ✅ models module structure")
        else:
            print("  ❌ models module structure")
        
        if Path('training/__init__.py').exists():
            print("  ✅ training module structure")
        else:
            print("  ❌ training module structure")
            
    except Exception as e:
        print(f"  ⚠️  Module structure test failed: {e}")

def test_configuration():
    """Test configuration and setup"""
    print("\n⚙️ Testing configuration...")
    
    # Test Python version
    python_version = sys.version_info
    print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 8):
        print("  ✅ Python version compatible")
    else:
        print("  ❌ Python version too old (need 3.8+)")
    
    # Test if we're in the right directory
    if Path('api.py').exists() and Path('app.py').exists():
        print("  ✅ In correct directory")
    else:
        print("  ❌ Not in correct directory")
    
    # Test demo launcher
    if Path('run_demo.py').exists():
        print("  ✅ Demo launcher available")
    else:
        print("  ❌ Demo launcher missing")

def main():
    """Main test function"""
    print("🧠 BrainWave Analyzer Setup Test")
    print("=" * 50)
    
    # Run tests
    dir_ok = test_directory_structure()
    file_ok = test_file_structure()
    test_module_imports()
    test_configuration()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    if dir_ok and file_ok:
        print("✅ Basic setup is correct!")
        print("\n🚀 Ready to run demo:")
        print("  python run_demo.py")
        print("\n📚 Or manually:")
        print("  python api.py        # Start API server")
        print("  streamlit run app.py # Start web interface")
    else:
        print("❌ Setup issues detected!")
        print("\n🔧 Please check:")
        print("  - All required files are present")
        print("  - All required directories exist")
        print("  - You're in the correct directory")
    
    print("\n💡 Note: This test doesn't check dependencies.")
    print("   Run 'python run_demo.py --check-only' to verify dependencies.")

if __name__ == "__main__":
    main()
