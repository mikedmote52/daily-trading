# CLAUDE.md - Explosive Stock Discovery Trading System

## 🚨 CRITICAL SYSTEM RULES - MUST FOLLOW

### SINGLE DISCOVERY SYSTEM ENFORCEMENT
**THE ONE AND ONLY DISCOVERY SYSTEM:** `/Users/michaelmote/Desktop/Daily-Trading/agents/discovery/universal_discovery.py`

**FORBIDDEN ACTIONS:**
- ❌ NEVER create new discovery files (no `fixed_universal_discovery.py`, `enhanced_discovery.py`, etc.)
- ❌ NEVER create backup/alternate discovery systems
- ❌ NEVER create "test" discovery systems that become permanent
- ❌ NEVER implement discovery logic anywhere else

**REQUIRED ACTIONS:**
- ✅ ALWAYS edit `universal_discovery.py` directly using Edit/MultiEdit tools
- ✅ ALWAYS check for duplicates before work: `find . -name "*discovery*.py"`
- ✅ ALWAYS remove any duplicate discovery files immediately upon discovery

## Project Overview

**Mission**: Build automated trading system to identify and trade stocks BEFORE explosive moves (targeting +63.8% returns)

**Target**: Pre-explosion accumulation patterns, NOT post-explosion chasing

**Current Implementation**: 
- Single universal discovery system with Gate A/B/C filtering
- Accumulation-based scoring (removed explosive detection)
- Polygon API for market data
- Alpaca API for trading execution

## System Architecture

### Directory Structure
```
/Users/michaelmote/Desktop/Daily-Trading/
├── agents/
│   ├── discovery/           # Discovery agent
│   │   └── universal_discovery.py  # THE ONLY DISCOVERY SYSTEM
│   ├── backend/            # FastAPI backend
│   ├── portfolio/          # Portfolio management
│   ├── backtesting/        # Historical validation
│   └── frontend/           # React UI
├── shared_context/         # Shared utilities
└── SYSTEM_RULES.md        # Critical system rules
```

## Discovery System Specifications

### Gate A Filtering (Initial Screen)
- Price: $0.01 - $100
- Volume: > 300,000 shares
- RVOL: > 1.3x (institutional interest)
- NO PERCENT CHANGE FILTER (removed explosive detection)

### Gate B Filtering (Fundamental)
- Market Cap: $100M - $50B
- ATR: > 4%
- Trend: Positive positioning

### Accumulation Scoring (NOT Explosive Scoring)
- Volume patterns (40% weight)
- Float/Short setup (30% weight)
- Options activity (20% weight)
- Technical positioning (10% weight)

## API Integrations

### Polygon API
- Market data and universe loading
- Real-time quotes and volumes
- NOT using Gainers API (removed)

### Alpaca API
- Paper and live trading execution
- Portfolio management
- Position tracking

### Claude AI
- Pattern recognition assistance
- Market analysis support

## Development Guidelines

### Before ANY Discovery Work:
1. Verify single system: `ls -la /Users/michaelmote/Desktop/Daily-Trading/agents/discovery/*.py`
2. Confirm only `universal_discovery.py` exists
3. Use Edit/MultiEdit tools ONLY
4. Never create new files

### Testing
- Use `test_universal.py` for testing
- Modify existing test file, don't create new ones
- Run in virtual environment: `source venv/bin/activate`

## Common Mistakes to Avoid

1. **Creating "fixed" versions** - Edit the original instead
2. **Using Gainers API** - This finds stocks AFTER explosion
3. **Filtering by percent change** - This biases toward already-moved stocks
4. **Creating duplicate systems** - Causes system failures and inconsistencies

## Recovery Protocol

If duplicate discovery systems are found:
1. Keep ONLY `/agents/discovery/universal_discovery.py`
2. Delete ALL other discovery implementations
3. Verify with: `find . -name "*discovery*.py" | grep -v venv`
4. Test single system functionality

---
**REMEMBER: ONE DISCOVERY SYSTEM. ONE SOURCE OF TRUTH. NO EXCEPTIONS.**