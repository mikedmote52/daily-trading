#!/usr/bin/env python3
"""
Simple Start Script for Portfolio API
Redirect to portfolio_api.py since Render expects this filename
"""

import os
import sys

def main():
    print("🔄 simple_start.py - Redirecting to portfolio_api.py")
    print("📁 Current directory:", os.getcwd())
    print("🐍 Python version:", sys.version)

    # Check environment
    print("\n🔧 Environment Check:")
    print(f"PORT: {os.environ.get('PORT', 'Not Set')}")
    print(f"ALPACA_KEY: {'✅ Set' if os.environ.get('ALPACA_KEY') else '❌ Missing'}")
    print(f"ALPACA_SECRET: {'✅ Set' if os.environ.get('ALPACA_SECRET') else '❌ Missing'}")
    print(f"ALPACA_BASE_URL: {os.environ.get('ALPACA_BASE_URL', 'Not Set')}")

    # Import and run the actual portfolio API
    print("\n🚀 Starting Portfolio API...")

    try:
        # Import the portfolio API app
        from portfolio_api import app
        import uvicorn

        port = int(os.environ.get("PORT", 8002))
        host = "0.0.0.0"

        print(f"🌐 Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    except Exception as e:
        print(f"❌ Failed to start portfolio API: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()