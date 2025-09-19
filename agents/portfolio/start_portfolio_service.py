#!/usr/bin/env python3
"""
Portfolio Service Startup Script

Starts both the portfolio manager and API server.
For production deployment on Render or similar platforms.
"""

import os
import sys
import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from enhanced_portfolio_manager import EnhancedPortfolioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PortfolioService')

# Global references
portfolio_manager = None
api_server = None
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False
    
    if portfolio_manager:
        portfolio_manager.running = False
    
    sys.exit(0)

async def start_portfolio_manager():
    """Start the portfolio manager in background"""
    global portfolio_manager
    
    try:
        logger.info("Starting Enhanced Portfolio Manager...")
        portfolio_manager = EnhancedPortfolioManager()
        
        # Initial portfolio load
        await portfolio_manager.update_portfolio()
        logger.info("Portfolio manager initialized successfully")
        
        # Start monitoring loops
        tasks = [
            asyncio.create_task(portfolio_manager._portfolio_monitoring_loop()),
            asyncio.create_task(portfolio_manager._health_analysis_loop()),
            asyncio.create_task(portfolio_manager._recommendation_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Error in portfolio manager: {e}")
        raise

def start_api_server():
    """Start the FastAPI server"""
    try:
        port = int(os.getenv("PORT", 8002))
        host = os.getenv("HOST", "0.0.0.0")
        
        logger.info(f"Starting Portfolio API server on {host}:{port}")
        
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        raise

async def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=== Enhanced Portfolio Management Service Starting ===")
    
    # Check required environment variables
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'CLAUDE_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return 1
    
    logger.info("Environment variables validated")
    
    try:
        # For production deployment, start API server directly
        # The portfolio manager will be started by the API server's lifespan
        
        if os.getenv('DEPLOYMENT_MODE') == 'production':
            logger.info("Production mode: Starting API server with integrated portfolio manager")
            start_api_server()
        else:
            logger.info("Development mode: Starting portfolio manager and API server concurrently")
            
            # Create executor for running API server in thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Start API server in thread
                api_future = executor.submit(start_api_server)
                
                # Start portfolio manager in current thread
                portfolio_future = asyncio.create_task(start_portfolio_manager())
                
                # Wait for either to complete
                done, pending = await asyncio.wait(
                    [portfolio_future],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                
                # Check if API server thread completed
                if api_future.done():
                    try:
                        api_future.result()
                    except Exception as e:
                        logger.error(f"API server error: {e}")
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error in portfolio service: {e}")
        return 1
    
    logger.info("Portfolio Management Service stopped")
    return 0

if __name__ == "__main__":
    # For production deployment (Render)
    if len(sys.argv) > 1 and sys.argv[1] == "--api-only":
        # Start only the API server (portfolio manager handled by lifespan)
        start_api_server()
    else:
        # Start full service
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
