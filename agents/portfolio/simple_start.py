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
    alpaca_key = os.environ.get('ALPACA_KEY', '')
    alpaca_secret = os.environ.get('ALPACA_SECRET', '')
    print(f"ALPACA_KEY: {'✅ Set' if alpaca_key else '❌ Missing'} (length: {len(alpaca_key)})")
    print(f"ALPACA_SECRET: {'✅ Set' if alpaca_secret else '❌ Missing'} (length: {len(alpaca_secret)})")
    print(f"ALPACA_BASE_URL: {os.environ.get('ALPACA_BASE_URL', 'Not Set')}")

    # Show first/last few characters of keys for debugging (without exposing full keys)
    if alpaca_key:
        print(f"ALPACA_KEY preview: {alpaca_key[:4]}...{alpaca_key[-4:] if len(alpaca_key) > 8 else ''}")
    if alpaca_secret:
        print(f"ALPACA_SECRET preview: {alpaca_secret[:4]}...{alpaca_secret[-4:] if len(alpaca_secret) > 8 else ''}")

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