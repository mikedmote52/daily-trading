# CLAUDE.md - Explosive Stock Discovery Trading System

## 🚨 CRITICAL SYSTEM RULES - MUST FOLLOW

### SINGLE DISCOVERY SYSTEM ENFORCEMENT
**THE ONE AND ONLY DISCOVERY SYSTEM:** `/agents/discovery/universal_discovery.py`

**FORBIDDEN ACTIONS:**
- ❌ NEVER create new discovery files (no `fixed_universal_discovery.py`, `enhanced_discovery.py`, etc.)
- ❌ NEVER create backup/alternate discovery systems
- ❌ NEVER create "test" discovery systems that become permanent
- ❌ NEVER implement discovery logic anywhere else

**REQUIRED ACTIONS:**
- ✅ ALWAYS edit `universal_discovery.py` directly using Edit/MultiEdit tools
- ✅ ALWAYS check for duplicates before work: `find agents/discovery -name "*discovery*.py"`
- ✅ ALWAYS remove any duplicate discovery files immediately upon discovery

### 100% REAL DATA POLICY
**ABSOLUTE RULE: Daily-Trading must NEVER use fake, mock, demo, template, or hardcoded data.**

**Forbidden Practices:**
- ❌ Template web enrichment (generic "Moderately bullish" sentiment)
- ❌ Mock candidates
- ❌ Hardcoded defaults
- ❌ Placeholder values
- ❌ Estimated/simulated data

**Required Practices:**
- ✅ Real-time Polygon API data ONLY
- ✅ Return `None` or empty results if data unavailable
- ✅ Fail with proper errors instead of fake data
- ✅ All calculations based on actual market data

## PRIMARY MISSION: REPLICATE +63.8% MONTHLY RETURNS

**Historical Performance Target (June 1 - July 4, 2024):**
- Portfolio: 15 positions × $100 = $1,500 initial capital
- Final value: $2,457.50
- Total return: +63.8% (+$957.50)
- Win rate: 93.3% (14/15 profitable)
- Top performer: VIGL +324% (detected at RVOL 1.8x, +0.4% price)
- Second: CRWV +171% (detected at RVOL 1.9x, -0.2% price)
- Third: AEVA +162% (detected at RVOL 1.7x, +1.1% price)
- Only loss: WOLF -25%

**System Goal:**
Identify explosive growth stocks BEFORE major moves using:
1. **Stealth accumulation detection** (RVOL 1.5-2.0x magic window)
2. **Pre-explosion positioning** (<2% price movement requirement)
3. **Multi-day pattern validation** (14-day sustained accumulation)
4. **Risk management** (prevent WOLF-like -25% losses)
5. **Portfolio structure** (15 positions, equal weighting)

**Key Insight:** VIGL/CRWV/AEVA were found BEFORE their explosions by detecting quiet institutional accumulation (high volume + stable price)

## DISCOVERY SYSTEM ARCHITECTURE

### Full Universe Scanning (REQUIRED)

**Correct Implementation:**
- Fetch FULL market snapshot - ALL 10,000+ stocks
- NO subset filtering, NO gainers/losers API
- Expected: 10,000+ stocks from full market

**FORBIDDEN (if found, remove immediately):**
- ❌ Using Gainers API (finds stocks AFTER explosion)
- ❌ Using Losers API
- ❌ Hardcoded symbol list
- ❌ Pre-filtering before Gate A

### Gate A: Initial Filtering
- Price: $5.00 - $100.00 (avoid penny stocks, within budget)
- Volume: > 300,000 shares (liquidity floor)
- RVOL: > 1.2x (institutional interest baseline)
- ETF Exclusion: Filter out ETFs and funds
- Blacklist: 30-day cooloff for recent losers

**Result:** ~500-2000 clean candidates

### Gate B: Stealth Detection (VIGL/CRWV/AEVA Pattern)

**The "Magic Window" (CRITICAL):**
- MIN_STEALTH_RVOL = 1.5 (✅ VIGL was 1.8x)
- MAX_STEALTH_RVOL = 2.0 (✅ Exclude already-discovered stocks)
- MAX_STEALTH_CHANGE = 2.0 (✅ VIGL was +0.4%, CRWV was -0.2%)
- MIN_PRICE = 5.0 (✅ Avoid penny stocks)

**Pattern Detection Logic:**
- High volume (1.5-2.0x RVOL) = Institutional accumulation
- Low price movement (<2%) = Stealth mode (not yet discovered)
- Price floor ($5+) = Quality stock filter
- Volume floor (300K+) = Liquidity requirement

**Historical Validation:**
- VIGL: 1.8x RVOL, +0.4% → PASSED → +324%
- CRWV: 1.9x RVOL, -0.2% → PASSED → +171%
- AEVA: 1.7x RVOL, +1.1% → PASSED → +162%

### Gate C: Accumulation Scoring (0-100 points)

**5-Component Scoring System:**

1. **Stealth Accumulation (35% weight)**
   - High volume + low price movement = institutional accumulation
   - Peak scoring at 1.5-2.0x RVOL
   - Reward price stability

2. **Sustained Pattern (20% weight)**
   - Multi-day accumulation (14-day history from Polygon)
   - Volume increasing over time
   - Real data only - return None if unavailable

3. **Small Cap Potential (20% weight)**
   - Lower price = higher explosion potential
   - VIGL was ~$2.50 at detection

4. **Coiling Pattern (15% weight)**
   - Volume increasing + volatility decreasing
   - Compression indicates imminent breakout

5. **Volume Quality (10% weight)**
   - RVOL closer to 2.0x = stronger accumulation
   - Institutional commitment signal

**VIGL Pattern Scoring:**
- Scores 75-85/100 for stocks matching VIGL characteristics
- Prioritizes pre-explosion stealth accumulation

### Adaptive Quality Gates

**Percentile-Based Selection:**
- Score threshold: Top 25% (75th percentile)
- RVOL threshold: Top 40% (60th percentile)
- Final filters: Volume 500K+, Price $1+, Change ≤8%
- Sort by accumulation score, select top 15

## PORTFOLIO MANAGEMENT (15 Positions × $100)

**Portfolio Structure:**
- Initial capital: $1,500
- Position size: $100 per stock
- Max positions: 15
- Target monthly return: 63.8%
- Stop loss: -15% max loss
- Trailing stop: -10% from peak

**Position Targets by Score:**
- Score ≥80: +200% target (VIGL-like)
- Score ≥70: +150% target (CRWV-like)
- Score ≥60: +100% target (AEVA-like)
- Score <60: +50% target

## RISK MANAGEMENT (Prevent WOLF Losses)

**Automatic Exit Triggers:**

1. **Stop Loss (-15%)**
   - Exit position if price drops 15%
   - Add to blacklist for 30-day cooloff
   - Prevents WOLF-like -25% losses

2. **Trailing Stop (-10% from peak)**
   - Activate when position is +20% profitable
   - Lock in profits as price rises
   - Exit if price drops 10% from peak

3. **Target Hit**
   - Exit when target price reached
   - Take profits on successful trades

**Blacklist Management:**
- Track recent losing stocks
- 30-day cooloff period
- Current blacklist: FATN, QMCO, QSI, NAK, PLTR, SOFI, GCCS, CDLX, CLOV, LAES

## DATA ENRICHMENT (100% Real Data)

**Real Data Sources:**
- Market cap & float: Polygon ticker details API
- Volume patterns: Polygon aggregates API
- Short interest: Polygon premium data
- Options flow: Polygon options API

**Removed Components:**
- ❌ Template web enrichment (was generating fake "Moderately bullish" for all stocks)
- ❌ Generic sentiment placeholders
- ❌ Mock data fallbacks

**Implementation:**
- Return None if data unavailable
- Retry logic with exponential backoff (2 retries, 10s timeout)
- Fail gracefully without fake data

## PERFORMANCE TRACKING

**Baseline Comparison:**
- Period: June 1 - July 4, 2024
- Total invested: $1,500
- Final value: $2,457.50
- Return: +63.8%
- Win rate: 93.3%
- Top performer: VIGL +324%
- Worst performer: WOLF -25%

**Performance Grading:**
- Grade A+: ≥63.8% return AND ≥93.3% win rate
- Grade A: ≥50% return OR ≥80% win rate
- Grade B: ≥30% return

## API INTEGRATIONS

### Polygon API
- Full market snapshot (10,000+ stocks)
- Real-time quotes and volumes
- Historical aggregates for sustained pattern analysis
- Ticker details for market cap and float
- Premium data: short interest, options flow

### Alpaca API
- Paper and live trading execution
- Portfolio management
- Position tracking
- Order management

## 🚀 MANDATORY DEPLOYMENT WORKFLOW - NEVER SKIP

### EVERY Code Change MUST Be Deployed:
1. **Make changes** and test locally
2. **Git add** the changed files: `git add <files>`
3. **Git commit** with descriptive message: `git commit -m "message"`
4. **Git push** to trigger deployment: `git push origin main`
5. **Confirm** push succeeded and inform user: "Changes pushed - Render will auto-deploy in ~10-15 minutes"

**⚠️ LOCAL CHANGES ARE WORTHLESS WITHOUT DEPLOYMENT**
**⚠️ ALWAYS COMPLETE THE FULL WORKFLOW**
**⚠️ RENDER ONLY DEPLOYS WHAT'S PUSHED TO GITHUB**

## CONFIGURATION

```bash
# .env
POLYGON_API_KEY=your_polygon_key_here

# Portfolio Settings
INITIAL_CAPITAL=1500
POSITION_SIZE_USD=100
MAX_POSITIONS=15
STOP_LOSS_PCT=0.15
TRAILING_STOP_PCT=0.10

# Stealth Detection (VIGL/CRWV/AEVA calibrated)
MIN_STEALTH_RVOL=1.5
MAX_STEALTH_RVOL=2.0
MAX_STEALTH_CHANGE=2.0
MIN_PRICE=5.0

# Performance Targets
TARGET_MONTHLY_RETURN=0.638
TARGET_WIN_RATE=0.933
```

## SUCCESS CRITERIA

✅ Discovery scans 10,000+ stocks (full universe)
✅ Only ONE discovery file (universal_discovery.py)
✅ RVOL magic window (1.5-2.0x) validated
✅ 15 positions selected, $100 each
✅ Win rate >90%
✅ Monthly return >60%
✅ No position loss >-15%
✅ At least one position >100% gain
✅ Performance grade: A or A+
✅ Template web enrichment REMOVED
✅ ZERO fake/mock/template data

## COMMON MISTAKES TO AVOID

1. **Creating "fixed" versions** - Edit the original instead
2. **Using Gainers API** - This finds stocks AFTER explosion
3. **Filtering by percent change** - This biases toward already-moved stocks
4. **Creating duplicate systems** - Causes system failures and inconsistencies
5. **FORGETTING TO PUSH CHANGES** - Local testing success ≠ deployment. ALWAYS push to GitHub!

## RECOVERY PROTOCOL

If duplicate discovery systems are found:
1. Keep ONLY `/agents/discovery/universal_discovery.py`
2. Delete ALL other discovery implementations
3. Verify with: `find . -name "*discovery*.py" | grep -v venv`
4. Test single system functionality

---
**REMEMBER: ONE DISCOVERY SYSTEM. ONE SOURCE OF TRUTH. NO EXCEPTIONS.**
