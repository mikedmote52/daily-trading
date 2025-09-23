# MCP Integration Guide for Daily Trading System

## Quick Setup

### 1. Environment Variables
```bash
export POLYGON_API_KEY="1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
export MCP_POLYGON_URL="https://polygon-mcp-server.onrender.com/mcp"
```

### 2. Three Integration Methods

#### Method A: Direct MCP Functions (Claude Code)
When running in Claude Code, use functions directly:
```python
# Short interest data
short_data = mcp__polygon__list_short_interest(ticker="AAPL", limit=1)

# Real-time snapshots
snapshot = mcp__polygon__get_snapshot_ticker(market_type="stocks", ticker="AAPL")

# Market-wide data
all_stocks = mcp__polygon__get_snapshot_all(market_type="stocks")
```

#### Method B: HTTP MCP Server (Render Deployment)
For your Render deployment, use the HTTP client in `mcp_integration_example.py`:
```python
client = PolygonMCPClient()
enhanced_data = await client.get_enhanced_data("AAPL")
```

#### Method C: Polygon API Fallback
Direct API calls when MCP unavailable:
```python
from polygon import RESTClient
client = RESTClient(api_key)
short_data = client.list_short_interest(ticker="AAPL")
```

## Available Enhanced Data

### ✅ Working with Current API Tier:
- **Short Interest**: `mcp__polygon__list_short_interest`
- **Real-time Snapshots**: `mcp__polygon__get_snapshot_ticker`
- **Market Overview**: `mcp__polygon__get_snapshot_all`
- **Historical Data**: `mcp__polygon__get_aggs`
- **Trades**: `mcp__polygon__list_trades`
- **Quotes**: `mcp__polygon__list_quotes`

### ⚠️ Requires Premium Tier:
- **Benzinga News**: `mcp__polygon__list_benzinga_news`
- **Analyst Insights**: `mcp__polygon__list_benzinga_analyst_insights`
- **Options Flow**: Advanced options data
- **Earnings**: `mcp__polygon__list_benzinga_earnings`

## Integration into Discovery System

### Step 1: Add Enhanced Data Collection
```python
def enrich_candidates_with_mcp(self, candidates: List[str]) -> Dict:
    \"\"\"Enrich discovery candidates with MCP data\"\"\"
    enhanced = {}

    for ticker in candidates:
        try:
            # Get short interest
            short_data = mcp__polygon__list_short_interest(
                ticker=ticker,
                limit=1
            )

            # Get real-time snapshot
            snapshot = mcp__polygon__get_snapshot_ticker(
                market_type="stocks",
                ticker=ticker
            )

            enhanced[ticker] = {
                'short_interest': short_data,
                'snapshot': snapshot,
                'score_boost': self._calculate_mcp_boost(short_data, snapshot)
            }

        except Exception as e:
            logger.warning(f"MCP enrichment failed for {ticker}: {e}")
            enhanced[ticker] = None

    return enhanced
```

### Step 2: Score Adjustments
```python
def _calculate_mcp_boost(self, short_data: Dict, snapshot: Dict) -> int:
    \"\"\"Calculate score boost from MCP data\"\"\"
    boost = 0

    # Short squeeze potential
    if short_data.get('results'):
        si = short_data['results'][0]
        days_to_cover = si.get('days_to_cover', 0)
        if days_to_cover > 3:
            boost += min(20, days_to_cover * 2)

    # Volume surge
    if snapshot.get('ticker'):
        prev_vol = snapshot['ticker']['prevDay'].get('v', 1)
        current_vol = snapshot['ticker']['min'].get('v', 0)
        if prev_vol > 0:
            vol_ratio = current_vol / (prev_vol / 390)  # Normalize to per-minute
            if vol_ratio > 2:
                boost += min(15, vol_ratio * 3)

    return boost
```

## Deployment Configuration

### Render Blueprint Addition
```yaml
# render.yaml
services:
  - type: web
    name: daily-trading-discovery
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python -m agents.discovery.universal_discovery"
    envVars:
      - key: POLYGON_API_KEY
        value: "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
      - key: MCP_POLYGON_URL
        value: "https://polygon-mcp-server.onrender.com/mcp"
```

## Testing Your Integration

Run the example:
```bash
cd /Users/michaelmote/Desktop/Daily-Trading
python mcp_integration_example.py
```

Expected output:
```
Testing MCP availability...
✅ Direct MCP functions available
{
  "short_interest": {
    "results": [...],
    "status": "OK"
  }
}

AAPL:
  Short Squeeze Potential: +4
  Options Momentum: +10
  Sentiment Score: +6
  Total Score Boost: +20
```

## Next Steps

1. **Test the integration**: Run `mcp_integration_example.py`
2. **Integrate into discovery**: Add MCP enrichment to `universal_discovery.py`
3. **Configure scoring**: Adjust boost calculations based on your strategy
4. **Deploy**: Update your Render configuration with MCP variables

The MCP provides real-time enhanced data that your current system can't access through basic API calls. This gives you competitive advantage in identifying pre-explosion setups.