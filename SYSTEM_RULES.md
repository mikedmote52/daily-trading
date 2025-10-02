# SYSTEM RULES - Daily-Trading

## CRITICAL RULE #1: SINGLE DISCOVERY SYSTEM

**THE ONLY DISCOVERY FILE:**
`/agents/discovery/universal_discovery.py`

**NEVER create:**
- fixed_universal_discovery.py
- enhanced_discovery.py
- discovery_v2.py
- new_discovery.py
- ANY other discovery file

**IF YOU FIND DUPLICATES:**
1. Stop immediately
2. Delete all duplicates
3. Keep ONLY universal_discovery.py
4. Verify with: `ls agents/discovery/*discovery*.py`

## CRITICAL RULE #2: 100% REAL DATA

**NEVER use:**
- Mock data
- Template data (like "Moderately bullish")
- Hardcoded values
- Placeholder data

**ALWAYS use:**
- Real Polygon API data
- Return None if unavailable
- Fail with errors, not fallbacks

## CRITICAL RULE #3: OPTIMIZATION MISSION

**Goal:** Replicate +63.8% monthly returns (June-July baseline)

**Target Metrics:**
- Monthly return: >60%
- Win rate: >90%
- Max loss: <-15%
- At least one >100% winner

**Portfolio Structure:**
- 15 positions
- $100 per position
- $1,500 total capital

## CRITICAL RULE #4: GIT WORKFLOW

**ALWAYS after code changes:**
1. git add .
2. git commit -m "descriptive message"
3. git push origin main
4. Wait 10-15 minutes for Render deployment

**NEVER skip git push** - local changes don't deploy

## CRITICAL RULE #5: VIGL PATTERN DETECTION

**Stealth Window:**
- RVOL: 1.5x to 2.0x (magic window)
- Price change: <2%
- Price: >$5

**Why:** VIGL (+324%), CRWV (+171%), AEVA (+162%) all found in this window BEFORE explosion

## CRITICAL RULE #6: NO GAINERS API

**NEVER use:**
- /v2/snapshot/locale/us/markets/stocks/gainers
- /v2/snapshot/locale/us/markets/stocks/losers

**Why:** Finds stocks AFTER explosion, not before

**ALWAYS use:**
- /v2/snapshot/locale/us/markets/stocks/tickers (full universe)

## CRITICAL RULE #7: BLACKLIST MANAGEMENT

**Current Blacklist (30-day cooloff):**
- FATN, QMCO, QSI, NAK, PLTR, SOFI
- GCCS, CDLX, CLOV, LAES

**Blacklist Triggers:**
- Position loss >15%
- Automatic 30-day exclusion
- Prevents repeated losses

## CRITICAL RULE #8: STEALTH DETECTION CONSTANTS

**Gate B Magic Window (DO NOT MODIFY without testing):**
```python
MIN_STEALTH_RVOL = 1.5    # Minimum for accumulation
MAX_STEALTH_RVOL = 2.0    # Maximum for stealth
MAX_STEALTH_CHANGE = 2.0  # Maximum price movement %
MIN_PRICE = 5.0           # Avoid penny stocks
```

**These values are calibrated to VIGL/CRWV/AEVA patterns**

## CRITICAL RULE #9: DEPLOYMENT VERIFICATION

**After every deployment, verify:**
1. Health check: `/health` endpoint responds
2. Watchlist: `/api/watchlist` returns candidates
3. RVOL window: All candidates have RVOL 1.5-2.0x
4. No template data: Check for "Moderately bullish" (should be None)

## CRITICAL RULE #10: NO FAKE DATA

**Forbidden:**
- Template enrichment
- Mock candidates
- Hardcoded sentiment
- Estimated values

**Required:**
- Polygon API only
- Return None on failure
- Real calculations only
- Fail loudly on errors
