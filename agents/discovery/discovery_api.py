#!/usr/bin/env python3
"""
FastAPI Backend for Explosive Stock Discovery System
Real-time API with WebSocket support for exceptional user experience
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import redis
import os
from contextlib import asynccontextmanager

from universal_discovery import UniversalDiscoverySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscoveryAPI")

# Global discovery system
discovery_system = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global discovery_system, redis_client

    logger.info("🚀 Initializing Explosive Discovery API...")

    # Initialize discovery system
    discovery_system = UniversalDiscoverySystem()

    # Initialize Redis if available
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None

    logger.info("✅ Discovery API ready for deployment")
    yield

    # Cleanup
    if redis_client:
        redis_client.close()

# FastAPI app with lifespan
app = FastAPI(
    title="Explosive Stock Discovery API",
    description="Real-time stock discovery system for pre-explosion patterns",
    version="2.0.1",
    lifespan=lifespan
)

# CORS middleware for frontend communication
ALLOWED_ORIGINS = [
    "https://alphastack-frontend.onrender.com",  # Render static URL
    "http://localhost:5173",                     # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# API Endpoints

@app.get("/test/polygon")
def test_polygon():
    """Test if Polygon API is working"""
    if not discovery_system:
        return {"error": "Discovery system not initialized"}

    api_key = discovery_system.polygon_api_key
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}

    try:
        import requests
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{yesterday}"
        params = {'apikey': api_key, 'adjusted': 'true'}
        response = requests.get(url, params=params, timeout=10)

        return {
            "polygon_configured": True,
            "api_key_length": len(api_key),
            "test_url": url,
            "status_code": response.status_code,
            "has_results": 'results' in response.json() if response.status_code == 200 else False,
            "result_count": len(response.json().get('results', [])) if response.status_code == 200 else 0
        }
    except Exception as e:
        return {
            "polygon_configured": True,
            "api_key_length": len(api_key),
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check for Render deployment"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "explosive-discovery-api",
        "version": "2.0.1"
    }

@app.get("/")
async def root():
    """API documentation"""
    return HTMLResponse("""
    <html>
        <head><title>Explosive Stock Discovery API</title></head>
        <body>
            <h1>🚀 Explosive Stock Discovery API</h1>
            <p>Real-time stock discovery system for pre-explosion patterns</p>
            <h2>Endpoints:</h2>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><code>POST /discover</code> - Run discovery scan</li>
                <li><code>GET /results/latest</code> - Get latest results</li>
                <li><code>WebSocket /ws</code> - Real-time updates</li>
            </ul>
        </body>
    </html>
    """)

@app.post("/discover")
async def run_discovery(background_tasks: BackgroundTasks):
    """
    Run explosive stock discovery scan
    Returns immediate response with scan ID, real results via WebSocket
    """
    if not discovery_system:
        raise HTTPException(status_code=500, detail="Discovery system not initialized")

    scan_id = f"scan_{int(datetime.now().timestamp())}"

    # Start background discovery task
    background_tasks.add_task(execute_discovery_scan, scan_id)

    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "Discovery scan initiated - results will stream via WebSocket",
        "timestamp": datetime.now().isoformat()
    }

async def execute_discovery_scan(scan_id: str):
    """Execute discovery scan and broadcast results"""
    try:
        logger.info(f"🔍 Starting discovery scan: {scan_id}")

        # Broadcast scan start
        await manager.broadcast(json.dumps({
            "type": "scan_started",
            "scan_id": scan_id,
            "timestamp": datetime.now().isoformat()
        }))

        # Run discovery
        result = discovery_system.run_universal_discovery()

        # Cache results if Redis available
        if redis_client:
            try:
                redis_client.setex(f"results:{scan_id}", 3600, json.dumps(result))
                redis_client.setex("results:latest", 3600, json.dumps(result))
            except Exception as e:
                logger.warning(f"Redis cache failed: {e}")

        # Broadcast results
        await manager.broadcast(json.dumps({
            "type": "scan_complete",
            "scan_id": scan_id,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }))

        logger.info(f"✅ Discovery scan complete: {scan_id}")

    except Exception as e:
        logger.error(f"Discovery scan failed: {e}")
        await manager.broadcast(json.dumps({
            "type": "scan_error",
            "scan_id": scan_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))

@app.get("/debug/gates")
def debug_gates():
    """Debug endpoint to see filtering at each gate"""
    if not discovery_system:
        raise HTTPException(status_code=500, detail="Discovery system not initialized")

    try:
        # Run just the universe ingestion and Gate A
        df = discovery_system.bulk_ingest_universe()
        logger.info(f"Universe ingested: {len(df)} stocks")

        if len(df) == 0:
            return {"error": "No universe data available"}

        # Run Gate A filtering
        gate_a_df = discovery_system.vectorized_gate_a(df)

        return {
            "universe_count": len(df),
            "gate_a_survivors": len(gate_a_df),
            "sample_universe": df.head(3).to_dict('records') if len(df) > 0 else [],
            "sample_gate_a": gate_a_df.head(3).to_dict('records') if len(gate_a_df) > 0 else [],
            "config": {
                "min_volume": discovery_system.config.GATEA_MIN_VOL,
                "min_rvol": discovery_system.config.GATEA_MIN_RVOL
            }
        }
    except Exception as e:
        logger.error(f"Debug gates error: {e}")
        return {"error": str(e)}

@app.get("/signals/top")
async def get_top_signals():
    """Get top discovery signals (thin wrapper endpoint)"""
    # Removed demo mode - using real data only

    if redis_client:
        try:
            cached_result = redis_client.get("results:latest")
            if cached_result:
                result = json.loads(cached_result)
                # Extract top signals if available
                if "final_recommendations" in result:
                    return {
                        "signals": result["final_recommendations"][:10],  # Top 10
                        "timestamp": result.get("timestamp"),
                        "metadata": {
                            "total_universe": result.get("universe_coverage", {}).get("total_universe", 0),
                            "final_count": len(result.get("final_recommendations", []))
                        }
                    }
        except Exception as e:
            logger.warning(f"Redis fetch failed: {e}")

    # Fallback: run fresh discovery
    try:
        result = discovery_system.run_universal_discovery()
        return {
            "signals": result.get("final_recommendations", [])[:10],
            "timestamp": result.get("timestamp"),
            "metadata": {
                "total_universe": result.get("universe_coverage", {}).get("total_universe", 0),
                "final_count": len(result.get("final_recommendations", []))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery system error: {str(e)}")

@app.get("/results/latest")
async def get_latest_results():
    """Get latest discovery results"""
    if redis_client:
        try:
            cached_result = redis_client.get("results:latest")
            if cached_result:
                return json.loads(cached_result)
        except Exception as e:
            logger.warning(f"Redis fetch failed: {e}")

    return {
        "message": "No recent results available",
        "suggestion": "Run POST /discover to generate new results"
    }

@app.get("/results/{scan_id}")
async def get_scan_results(scan_id: str):
    """Get specific scan results"""
    if redis_client:
        try:
            cached_result = redis_client.get(f"results:{scan_id}")
            if cached_result:
                return json.loads(cached_result)
        except Exception as e:
            logger.warning(f"Redis fetch failed: {e}")

    raise HTTPException(status_code=404, detail="Scan results not found")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send welcome message
        await manager.send_personal_message(json.dumps({
            "type": "connected",
            "message": "Connected to Explosive Discovery API",
            "timestamp": datetime.now().isoformat()
        }), websocket)

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle client requests
                if message.get("type") == "ping":
                    await manager.send_personal_message(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }), websocket)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "active_connections": len(manager.active_connections),
        "redis_available": redis_client is not None,
        "discovery_system_ready": discovery_system is not None,
        "uptime": "Available via health endpoint",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "discovery_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )