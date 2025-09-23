# System Flow and Data Pathways Documentation

## Complete System Data Flow

### 1. Discovery System Flow

#### A. Universe Loading Pipeline
```
START → Environment Detection → MCP Availability Check
    ↓
Direct MCP (Claude Code):
    mcp__polygon__get_snapshot_all(market_type="stocks")
    → 8000+ stock universe → Gate A Processing

HTTP MCP (Render):
    HttpMcpClient.call_function("get_snapshot_all")
    → JSON response → Parse to stock data → Gate A Processing

Fallback (Direct API):
    polygon.RESTClient.get_snapshot_all()
    → API response → Parse to stock data → Gate A Processing
```

#### B. Gate A Filtering (Initial Screen)
```
Stock Universe (8000+ stocks)
    ↓
Filter Criteria:
    - Price: $0.01 - $100
    - Volume: > 300,000 shares
    - RVOL: > 1.3x (institutional interest)
    - NO percent change filter (anti-chase)
    ↓
Filtered Universe (~500-1000 stocks) → Gate B
```

#### C. Gate B Filtering (Fundamentals)
```
Gate A Survivors (~500-1000 stocks)
    ↓
Enhanced Data Enrichment:
    - Market Cap: $100M - $50B
    - ATR: > 4%
    - Trend: Positive positioning
    - Short Interest (via MCP): Days to cover analysis
    ↓
Gate B Survivors (~50-100 stocks) → Gate C Scoring
```

#### D. Gate C Accumulation Scoring
```
Gate B Survivors (~50-100 stocks)
    ↓
Scoring Algorithm (Accumulation-Based):
    - Volume Patterns: 40% weight
    - Float/Short Setup: 30% weight
    - Options Activity: 20% weight (when available)
    - Technical Positioning: 10% weight
    ↓
Scored Results (Top 20) → Cache → Backend API
```

### 2. MCP Data Flow Architecture

#### A. MCP Function Resolution Flow
```
Function Call Request
    ↓
Environment Detection:
    1. Claude Code: globals()['mcp__polygon__*'] → Direct call
    2. Render: try mcp__polygon__* → HTTP fallback
    3. Local: builtins check → API fallback
    ↓
Execution Path Selection:
    - Direct MCP: Native function call
    - HTTP MCP: FastMCP protocol via aiohttp
    - API Fallback: Direct Polygon REST client
    ↓
Response Processing → Cache → Return to caller
```

#### B. MCP Caching Strategy
```
Request → Cache Key Generation (f"{ticker}_{function}_{timestamp}")
    ↓
Cache Check (Redis/Memory):
    - Hit: Return cached data (if < 1 hour old)
    - Miss: Execute MCP call → Store in cache → Return data
    ↓
Cache Expiration: 1 hour for most data, 5 minutes for real-time
```

### 3. Backend API Data Flow

#### A. FastAPI Request Processing
```
HTTP Request → CORS Middleware → Route Handler
    ↓
Request Types:
    - GET /stocks/explosive → Discovery pipeline trigger
    - GET /portfolio/positions → Alpaca API call
    - GET /trades/history → Database/Redis query
    - WebSocket /ws → Real-time updates
    ↓
Response → JSON serialization → CORS headers → Client
```

#### B. Inter-Agent Communication
```
Discovery Agent → Redis Message Queue
    ↓
Message Types:
    - explosive_opportunity: {symbol, score, reasoning}
    - market_update: {timestamp, status, volatility}
    - system_alert: {level, message, component}
    ↓
Portfolio Agent → Backtesting Agent → Frontend Updates
```

### 4. Portfolio Management Flow

#### A. Trade Decision Pipeline
```
Discovery Signals → Risk Analysis
    ↓
Risk Metrics Calculation:
    - Position Size: Kelly Criterion modified
    - Portfolio Impact: Correlation analysis
    - Risk Limits: Max 10% per position, 30% per sector
    ↓
Alpaca API Call → Order Execution → Position Tracking
```

#### B. Position Monitoring Flow
```
Real-time Price Updates (WebSocket)
    ↓
Position Valuation:
    - Unrealized P&L calculation
    - Risk metric updates (VaR, Sharpe)
    - Stop-loss trigger checks
    ↓
Portfolio Rebalancing Signals → Trade Execution
```

### 5. Frontend Data Flow

#### A. React Component Data Flow
```
Component Mount → useEffect Hook → API Call
    ↓
Data Fetching:
    - axios.get('/stocks/explosive') → Stock discovery results
    - socket.on('portfolio_update') → Real-time positions
    - useQuery() → React Query caching
    ↓
State Updates → Component Re-render → UI Display
```

#### B. Real-time Updates Flow
```
Backend Event → WebSocket Broadcast
    ↓
Frontend WebSocket Handler:
    - portfolio_update → Update positions state
    - market_alert → Show notification
    - new_opportunity → Add to discovery list
    ↓
React State Update → Component Re-render
```

## Data Storage and Persistence

### 1. Redis Cache Architecture
```
Cache Keys:
    - stock_data:{symbol}:{timestamp}
    - portfolio_positions:{account_id}
    - discovery_results:{scan_id}
    - mcp_cache:{function}:{params_hash}

Expiration Strategy:
    - Real-time data: 5 minutes
    - Enhanced data: 1 hour
    - Portfolio data: 15 minutes
    - Discovery results: 30 minutes
```

### 2. File System Storage
```
/agents/discovery/
    - universal_discovery.py: Core discovery logic
    - models.py: Data structure definitions
    - config.py: Configuration management

/shared_context/
    - mission_directive.json: System objectives
    - progress.json: Current system state
    - messages.json: Inter-agent communication

/logs/
    - discovery.log: Discovery system activity
    - trading.log: Execution and portfolio changes
    - system.log: Overall system health
```

## Error Handling and Recovery

### 1. MCP Error Recovery Flow
```
MCP Function Call → Exception Handling
    ↓
Error Types:
    - Connection Timeout: Retry with exponential backoff
    - Rate Limit: Wait and retry with delay
    - Service Unavailable: Fallback to next tier
    - Data Error: Log and use cached data
    ↓
Fallback Sequence:
    Direct MCP → HTTP MCP → Polygon API → Cached Data → Mock Data
```

### 2. System Health Monitoring
```
Component Health Checks (Every 60 seconds):
    - Discovery System: Last successful scan timestamp
    - MCP Connection: Ping test with timeout
    - Portfolio Service: Alpaca API connectivity
    - Database: Redis connection test
    ↓
Health Status → System Dashboard → Alert Generation
```

## Performance Optimization Pathways

### 1. Data Flow Optimization
```
Batch Processing:
    - Single get_snapshot_all() → 8000 stocks vs 8000 individual calls
    - Concurrent MCP calls using asyncio.gather()
    - Vectorized pandas operations for filtering

Caching Strategy:
    - L1 Cache: In-memory for current scan (5 minutes)
    - L2 Cache: Redis for enhanced data (1 hour)
    - L3 Cache: File system for historical patterns (1 day)
```

### 2. Latency Reduction
```
Critical Path Analysis:
    - Universe Loading: 2-5 seconds (cached: 100ms)
    - Gate A Processing: 500ms (vectorized)
    - Gate B Enhancement: 2-3 seconds (concurrent)
    - Gate C Scoring: 200ms (optimized algorithms)

Total Discovery Latency:
    - Cold Start: 5-8 seconds
    - Warm Cache: 1-2 seconds
    - Hot Cache: 200-500ms
```

## Security and Access Control

### 1. API Security Flow
```
External Request → Rate Limiting → Authentication Check
    ↓
API Key Validation:
    - Polygon API: Bearer token validation
    - Alpaca API: OAuth token refresh
    - Internal APIs: JWT token verification
    ↓
Request Processing → Response Encryption → Client
```

### 2. Data Protection
```
Sensitive Data Handling:
    - API Keys: Environment variables only
    - Trading Data: Encrypted at rest
    - User Data: Hashed and salted
    - Logs: PII scrubbed automatically
```

## Integration Points

### 1. External API Integration
```
Polygon API:
    - Market Data: REST endpoints for historical data
    - Real-time: WebSocket for live updates
    - Enhanced: MCP server for premium features

Alpaca API:
    - Trading: REST for order management
    - Portfolio: Real-time position updates
    - Market Data: Alternative data source

Anthropic API:
    - Pattern Analysis: Claude for market insights
    - Risk Assessment: AI-driven decision support
```

### 2. Internal Service Communication
```
Message Queue Pattern (Redis):
    - Publisher: Discovery service
    - Subscribers: Portfolio, Backtesting, Frontend
    - Message Types: JSON with schema validation

API Gateway Pattern:
    - Central FastAPI backend
    - Route-based service delegation
    - Unified authentication and logging
```

## Deployment Data Flow

### 1. Render.com Architecture
```
GitHub Push → Render Build → Container Deploy
    ↓
Service Mesh:
    - Discovery Service: Python container with MCP client
    - Backend Service: FastAPI with Redis connection
    - Frontend Service: React build with nginx
    ↓
External Dependencies:
    - Polygon MCP Server: polygon-mcp-server.onrender.com
    - Redis: Render-managed instance
    - Environment Variables: Render dashboard
```

### 2. Local Development Flow
```
Development Environment:
    - Docker Compose: Multi-service orchestration
    - Local Redis: Development cache
    - Mock Services: MCP fallback for testing
    ↓
Testing Pipeline:
    - Unit Tests: Individual component testing
    - Integration Tests: Service-to-service testing
    - End-to-end Tests: Full pipeline validation
```

## Monitoring and Observability

### 1. Metrics Collection
```
System Metrics:
    - Discovery Performance: Scan time, success rate, candidates found
    - MCP Performance: Response time, error rate, cache hit rate
    - Trading Performance: Fill rate, slippage, P&L
    - System Health: CPU, memory, disk, network

Logging Strategy:
    - Structured JSON logs
    - Correlation IDs for request tracing
    - Multi-level logging (DEBUG, INFO, WARN, ERROR)
    - Log aggregation and analysis
```

### 2. Alerting System
```
Alert Triggers:
    - Discovery Failures: No successful scan in 10 minutes
    - MCP Connectivity: Connection failures > 5 per minute
    - Trading Errors: Order rejections or API errors
    - Performance Degradation: Response time > 10 seconds

Notification Channels:
    - Real-time: WebSocket to frontend dashboard
    - Email: Critical system failures
    - Slack: Operational alerts
    - SMS: Emergency escalation
```

This comprehensive flow documentation shows how your Daily Trading System processes data from initial market scanning through final trade execution, with sophisticated MCP integration providing enhanced data capabilities throughout the pipeline.