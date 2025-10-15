#!/usr/bin/env python3
"""
BrainWave Analyzer Demo Launcher
One-command demo setup and execution
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
import signal
from pathlib import Path
import argparse


class BrainWaveDemoLauncher:
    """Launcher for BrainWave Analyzer demo"""
    
    def __init__(self):
        self.api_process = None
        self.streamlit_process = None
        self.running = False
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'tensorflow', 'streamlit', 'fastapi', 'uvicorn',
            'plotly', 'pandas', 'numpy', 'opencv-python-headless',
            'pillow', 'scipy', 'requests'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package}")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Installing missing dependencies...")
            
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages)
                print("✅ Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                return False
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Setting up directories...")
        
        directories = [
            'data', 'models', 'training', 'evaluation', 
            'utils', 'checkpoints', 'logs', 'visualizations'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"  ✅ {directory}/")
    
    def start_api_server(self, port=8000):
        """Start FastAPI server"""
        print(f"🚀 Starting API server on port {port}...")
        
        try:
            self.api_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'api:app', '--host', '0.0.0.0', '--port', str(port),
                '--reload'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if server is running
            if self.api_process.poll() is None:
                print(f"✅ API server started successfully on http://localhost:{port}")
                return True
            else:
                print("❌ Failed to start API server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def start_streamlit_app(self, port=8501):
        """Start Streamlit application"""
        print(f"🌐 Starting Streamlit app on port {port}...")
        
        try:
            self.streamlit_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', 'app.py',
                '--server.port', str(port), '--server.headless', 'true'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for app to start
            time.sleep(5)
            
            # Check if app is running
            if self.streamlit_process.poll() is None:
                print(f"✅ Streamlit app started successfully on http://localhost:{port}")
                return True
            else:
                print("❌ Failed to start Streamlit app")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Streamlit app: {e}")
            return False
    
    def open_browser(self, url):
        """Open browser to the application"""
        print(f"🌐 Opening browser to {url}...")
        try:
            webbrowser.open(url)
            print("✅ Browser opened successfully!")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"Please manually open: {url}")
    
    def monitor_processes(self):
        """Monitor running processes"""
        print("\n📊 Monitoring processes...")
        print("Press Ctrl+C to stop all services")
        
        try:
            while self.running:
                # Check API process
                if self.api_process and self.api_process.poll() is not None:
                    print("⚠️  API server stopped unexpectedly")
                    self.running = False
                    break
                
                # Check Streamlit process
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    print("⚠️  Streamlit app stopped unexpectedly")
                    self.running = False
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            self.running = False
    
    def stop_services(self):
        """Stop all running services"""
        print("🛑 Stopping services...")
        
        if self.api_process:
            self.api_process.terminate()
            print("✅ API server stopped")
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            print("✅ Streamlit app stopped")
    
    def run_demo(self, api_port=8000, streamlit_port=8501, open_browser=True):
        """Run the complete demo"""
        print("🧠 BrainWave Analyzer Demo Launcher")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed. Please install missing packages.")
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Start services
        if not self.start_api_server(api_port):
            return False
        
        if not self.start_streamlit_app(streamlit_port):
            self.stop_services()
            return False
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            self.stop_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.running = True
        
        # Open browser
        if open_browser:
            self.open_browser(f"http://localhost:{streamlit_port}")
        
        # Display status
        print("\n🎉 BrainWave Analyzer is running!")
        print("=" * 50)
        print(f"📊 API Server: http://localhost:{api_port}")
        print(f"🌐 Web Interface: http://localhost:{streamlit_port}")
        print(f"📚 API Documentation: http://localhost:{api_port}/docs")
        print("\n🎯 Features Available:")
        print("  • EEG → Image synthesis")
        print("  • Image → EEG prediction")
        print("  • Interactive visualizations")
        print("  • Real-time signal analysis")
        print("  • Download generated content")
        
        # Monitor processes
        self.monitor_processes()
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BrainWave Analyzer Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py                    # Run with default settings
  python run_demo.py --no-browser       # Don't open browser automatically
  python run_demo.py --api-port 8001    # Use custom API port
  python run_demo.py --streamlit-port 8502  # Use custom Streamlit port
        """
    )
    
    parser.add_argument(
        '--api-port', 
        type=int, 
        default=8000,
        help='Port for API server (default: 8000)'
    )
    
    parser.add_argument(
        '--streamlit-port', 
        type=int, 
        default=8501,
        help='Port for Streamlit app (default: 8501)'
    )
    
    parser.add_argument(
        '--no-browser', 
        action='store_true',
        help='Don\'t open browser automatically'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true',
        help='Only check dependencies and setup, don\'t start services'
    )
    
    args = parser.parse_args()
    
    launcher = BrainWaveDemoLauncher()
    
    if args.check_only:
        print("🔍 Checking setup only...")
        if launcher.check_dependencies():
            launcher.setup_directories()
            print("✅ Setup check completed successfully!")
        else:
            print("❌ Setup check failed.")
            sys.exit(1)
    else:
        success = launcher.run_demo(
            api_port=args.api_port,
            streamlit_port=args.streamlit_port,
            open_browser=not args.no_browser
        )
        
        if not success:
            print("❌ Demo failed to start.")
            sys.exit(1)


if __name__ == "__main__":
    main()
