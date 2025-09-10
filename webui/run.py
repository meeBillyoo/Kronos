#!/usr/bin/env python3
"""
Kronos Web UI startup script
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import plotly
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Dependencies installation failed")
        return False

def main():
    """Main function"""
    # 从环境变量获取端口，默认为 8068
    port = int(os.environ.get('PORT', 8068))
    
    print("🚀 Starting Kronos Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return
    
    # Check model availability
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos model library available")
        model_available = True
    except ImportError:
        print("⚠️  Kronos model library not available, will use simulated prediction")
        model_available = False
    
    # Start Flask application
    print("\n🌐 Starting Web server...")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Start server
    try:
        from app import app
        print("✅ Web server started successfully!")
        print(f"🌐 Main page: http://localhost:{port}")
        print(f"🌐 加密货币页面: http://localhost:{port}/crypto.html")
        print("💡 Tip: Press Ctrl+C to stop server")
        
        # Auto-open browser with option to choose page
        print("\nWhich page would you like to open?")
        print("1. Main page (index.html)")
        print("2. 加密货币交易所页面 (crypto.html)")
        print("3. Don't open browser")
        
        choice = input("Enter your choice (1-3, default: 1): ").strip()
        
        time.sleep(2)
        if choice == '2':
            webbrowser.open(f'http://localhost:{port}/crypto.html')
        elif choice == '3':
            pass  # Don't open browser
        else:
            webbrowser.open(f'http://localhost:{port}')
        
        # Start Flask application with environment variable port
        app.run(debug=True, host='0.0.0.0', port=port)
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print(f"Please check if port {port} is occupied")

if __name__ == "__main__":
    main()
