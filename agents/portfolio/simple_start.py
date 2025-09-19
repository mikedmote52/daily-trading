#!/usr/bin/env python3
"""
Simple Portfolio Service Startup
Direct uvicorn startup for production deployment
"""

import os
import uvicorn
from dotenv import load_dotenv

# Load environment
load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Portfolio API on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
