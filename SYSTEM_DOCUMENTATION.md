# 🚀 ALPHASTACK TRADING SYSTEM - COMPLETE DOCUMENTATION
**Version: 2.0.1 STABLE** - Lock this version as fallback
**Date: September 18, 2025**
**Status: FULLY OPERATIONAL**

---

## 📊 SYSTEM OVERVIEW

### Mission
**Find stocks BEFORE explosive moves (+63.8% returns)**
- Target: Pre-explosion accumulation patterns
- Method: Volume surge detection + technical analysis
- Current Success: 8 A-Tier candidates with 58x-634x volume surges

### Core Components
1. **Discovery Engine** - Finds pre-explosion opportunities
2. **Trading API** - Executes trades via Alpaca
3. **Frontend UI** - React interface for monitoring
4. **Deployment** - Render.com production hosting

---

## 🏗️ SYSTEM ARCHITECTURE

```
Daily-Trading/
├── agents/
│   ├── discovery/
│   │   ├── universal_discovery.py  # MAIN DISCOVERY ENGINE
│   │   ├── discovery_api.py        # FastAPI endpoints
│   │   ├── Dockerfile              # Container config
│   │   ├── requirements.txt        # Python deps
│   │   └── startup.sh              # Launch script
│   ├── backend/
│   │   ├── orders_api.py           # Alpaca integration
│   │   └── Dockerfile
│   └── portfolio/
│       └── portfolio_manager.py    # Position management
├── frontend/
│   ├── src/
│   │   └── main.jsx                # React UI
│   ├── package.json
│   └── vite.config.js
├── render.yaml                      # Deployment config
└── CLAUDE.md                        # System rules
```

---

## 🔬 DISCOVERY PIPELINE

### Complete Flow: Universe → Gate A → Gate B → Gate C → Final

**Current Performance:**
- Universe: 11,199 stocks (all US equities)
- Gate A: 6,106 survivors (54.5% pass rate)
- Gate B: 5,909 survivors (96.8% pass rate)
- Gate C: 8 final candidates (0.14% pass rate)

### Gate A - Initial Filtering (Rebuilt & Working)
```python
# Simple, reliable filters
- Price: $1.00 - $500.00
- Volume: > 50,000 shares
- Symbol: 1-5 letters, alphabetic only
```

### Gate B - Fundamental Analysis
```python
# Momentum and volatility
- ATR: ≥ 4% (adequate trading range)
- Trend: Positive 3-day momentum
# Note: Market cap filtering disabled (data unavailable)
```

### Gate C - Final Scoring & Selection
```python
# Multi-stage filtering:
1. Sustained RVOL (≥3x for 30+ minutes)
2. VWAP reclaim (price > VWAP)
3. EMA crossover (9 EMA > 20 EMA)
4. Liquidity check ($1M+ turnover)
5. Accumulation scoring (0-100 scale)
6. A-Tier selection (≥85 composite score)
```

---

## 📈 CURRENT DISCOVERIES (Live)

| Rank | Symbol | Price  | Volume Surge | Trend  | Score |
|------|--------|--------|--------------|--------|-------|
| 1    | QBTS   | $22.54 | 634x         | +18.8% | 91.6  |
| 2    | CIFR   | $12.38 | 170x         | +9.6%  | 91.6  |
| 3    | BE     | $79.67 | 82x          | +8.0%  | 91.6  |
| 4    | RCAT   | $11.27 | 65x          | +8.0%  | 91.6  |
| 5    | RGTI   | $21.99 | 389x         | +9.6%  | 91.6  |
| 6    | EXK    | $6.50  | 96x          | +7.1%  | 91.6  |
| 7    | EDUC   | $1.80  | 58x          | +16.1% | 91.6  |
| 8    | UP     | $2.19  | 68x          | +7.4%  | 91.6  |

---

## 🌐 API ENDPOINTS

### Discovery Service (https://alphastack-discovery.onrender.com)
- `GET /health` - Service health check
- `GET /signals/top` - Get top trade opportunities ✅
- `POST /discover` - Run fresh discovery scan
- `GET /debug/gates` - Gate-by-gate analysis
- `GET /debug/environment` - Environment info

### Orders Service (https://alphastack-orders.onrender.com)
- `GET /health` - Service health check
- `POST /orders/buy` - Execute buy order via Alpaca
- `POST /orders/sell` - Execute sell order
- `GET /portfolio` - Get current positions

---

## 🔧 CONFIGURATION

### Environment Variables
```bash
# Discovery Service
POLYGON_API_KEY=****  # Market data access
REDIS_URL=****        # Cache for results
DISCOVERY_CYCLE_SECONDS=60
TOP_N=8

# Orders Service  
ALPACA_KEY=****       # Trading API key
ALPACA_SECRET=****    # Trading API secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Key Thresholds
- Min Volume: 50,000 shares
- Min ATR: 4%
- Min RVOL: 3x sustained
- Min Liquidity: $1M turnover
- A-Tier Score: ≥85

---

## 🚨 CRITICAL RULES

1. **ONE DISCOVERY SYSTEM ONLY**
   - File: `/agents/discovery/universal_discovery.py`
   - Never create duplicates or alternatives
   - Always edit directly, never replace

2. **ACCUMULATION FOCUS**
   - Target pre-explosion patterns
   - NOT post-explosion chasing
   - Volume surge is key signal

3. **PRODUCTION SAFETY**
   - Always test locally first
   - Git commit triggers deployment
   - Render auto-deploys from main branch

---

## 🔄 VERSION CONTROL

### Creating Stable Checkpoint
```bash
# Tag current stable version
git tag -a v2.0.1-stable -m "Stable discovery system - 8 A-Tier stocks working"
git push origin v2.0.1-stable

# To revert if needed:
git checkout v2.0.1-stable
```

### Key Commits
- `dfaa0407` - Final API fixes for real discovery data
- `c43177fb` - Complete pipeline flow A→B→C→Final
- `efcd01ef` - Rebuilt Gate A with simple filtering

---

## 🐛 KNOWN ISSUES & FIXES NEEDED

1. **Buy Button Integration** ❌
   - Current: Buy button not connected to Alpaca
   - Needed: Frontend → Orders API → Alpaca flow
   
2. **Investment Thesis Missing** ❌
   - Current: Only shows score and metrics
   - Needed: AI-generated thesis with price targets

3. **Market Cap Data** ⚠️
   - Current: All stocks show market_cap=None
   - Impact: Can't filter by company size
   - Workaround: Disabled market cap filters

---

## 📝 TESTING COMMANDS

### Local Testing
```bash
# Activate environment
source venv/bin/activate

# Test discovery
python test_local_discovery.py

# Debug gates
python debug_gates.py
```

### API Testing
```bash
# Check health
curl https://alphastack-discovery.onrender.com/health

# Get signals
curl https://alphastack-discovery.onrender.com/signals/top
```

---

## 🎯 NEXT DEVELOPMENT PRIORITIES

1. **Alpaca Buy Integration**
   - Connect buy button to orders API
   - Implement position sizing logic
   - Add order confirmation flow

2. **Investment Thesis Generation**
   - Add thesis field to discovery results
   - Calculate price targets (e.g., +63.8% from entry)
   - Generate reasoning for each pick

3. **Performance Tracking**
   - Track actual vs projected returns
   - Monitor success rate over time
   - Build confidence metrics

---

## 📞 SUPPORT

- GitHub: https://github.com/mikedmote52/daily-trading
- Render Dashboard: https://dashboard.render.com
- Polygon API: https://polygon.io
- Alpaca API: https://alpaca.markets

---

**END OF DOCUMENTATION - SYSTEM STABLE AND OPERATIONAL**