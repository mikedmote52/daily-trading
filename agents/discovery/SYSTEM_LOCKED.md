# 🔒 SYSTEM LOCKED - SINGLE DISCOVERY FILE ENFORCEMENT

## CRITICAL: This directory is LOCKED to a single discovery system

**Date Locked:** October 1, 2025
**Enforcement Level:** MAXIMUM

---

## THE ONE AND ONLY DISCOVERY SYSTEM

**File:** `universal_discovery.py`
**Status:** ✅ ACTIVE AND LOCKED
**Location:** `/agents/discovery/universal_discovery.py`

---

## 🚫 STRICTLY FORBIDDEN ACTIONS

**NEVER create any of these files:**
- ❌ `fixed_universal_discovery.py`
- ❌ `enhanced_discovery.py`
- ❌ `discovery_v2.py`
- ❌ `improved_discovery.py`
- ❌ `new_discovery_system.py`
- ❌ `test_discovery.py` (as a permanent system)
- ❌ ANY other discovery implementation file

**Why This Rule Exists:**
1. Multiple discovery systems cause system failures
2. Competing implementations create inconsistent results
3. Wasted computational resources from duplicate processing
4. Confusion about which system is authoritative
5. Deployment conflicts and version control issues

---

## ✅ REQUIRED ACTIONS ONLY

**To modify the discovery system:**
1. ✅ Edit `universal_discovery.py` directly using Edit/MultiEdit tools
2. ✅ Test changes in the existing file
3. ✅ Never create a "new" version or "fixed" version
4. ✅ Always verify single system: `ls -la *discovery*.py`

**Before ANY discovery work:**
```bash
# Verify only ONE discovery file exists
ls -la /Users/michaelmote/Desktop/Daily-Trading/agents/discovery/*discovery*.py

# Expected output: ONLY universal_discovery.py
```

---

## 🔍 DUPLICATE DETECTION PROTOCOL

**If you discover duplicate discovery files:**

1. **STOP IMMEDIATELY** - Do not proceed with any work
2. **Identify all discovery files:**
   ```bash
   find /Users/michaelmote/Desktop/Daily-Trading -name "*discovery*.py" -type f | grep -v venv
   ```
3. **Keep ONLY `universal_discovery.py`**
4. **Delete ALL other discovery implementations:**
   ```bash
   # Example (verify first!)
   rm agents/discovery/fixed_universal_discovery.py
   rm agents/discovery/enhanced_discovery.py
   ```
5. **Verify single system remains:**
   ```bash
   ls -la agents/discovery/*discovery*.py
   # Should show ONLY: universal_discovery.py
   ```
6. **Document the incident** in git commit message

---

## 📊 CURRENT OPTIMIZATION MISSION

**Goal:** Replicate +63.8% monthly returns (June-July 2024 baseline)

**Key Metrics:**
- Portfolio: 15 positions × $100 = $1,500
- Target return: >60% monthly
- Target win rate: >90%
- Max loss per position: <-15%

**Stealth Detection Window (VIGL/CRWV/AEVA Pattern):**
- RVOL: 1.5x - 2.0x (magic window)
- Price change: <2% (pre-explosion positioning)
- Price: >$5 (quality filter)
- Pattern: 14-day sustained accumulation

**Historical Winners Found in This Window:**
- VIGL: +324% (RVOL 1.8x, +0.4% change)
- CRWV: +171% (RVOL 1.9x, -0.2% change)
- AEVA: +162% (RVOL 1.7x, +1.1% change)

---

## 🛡️ ENFORCEMENT CHECKLIST

Before committing any discovery-related changes:

- [ ] Verified only `universal_discovery.py` exists
- [ ] No duplicate discovery files created
- [ ] Changes made to existing file only (using Edit tool)
- [ ] Ran: `find . -name "*discovery*.py" | wc -l` (result should be 1)
- [ ] Tested changes in the single system
- [ ] Ready to commit and deploy

---

## 🚨 VIOLATION CONSEQUENCES

**What happens if this rule is violated:**
- System failures and crashes
- Inconsistent discovery results
- Production deployment failures
- Data inconsistencies
- Wasted API calls and resources
- Confusion about system state

**Recovery Cost:**
- Time lost identifying the issue
- Manual cleanup of duplicate systems
- Re-testing and re-deployment
- Potential loss of trading opportunities

---

## ✅ SUCCESS CRITERIA

**Single System Enforcement:**
- ✅ Only ONE file exists: `universal_discovery.py`
- ✅ All modifications via Edit/MultiEdit tools
- ✅ No duplicate systems in git history
- ✅ Clean deployment pipeline
- ✅ Consistent system behavior

**Performance Targets:**
- ✅ Monthly return: >60%
- ✅ Win rate: >90%
- ✅ 15 positions, $100 each
- ✅ At least one >100% winner per month
- ✅ No position loss >-15%

---

**REMEMBER: ONE DISCOVERY SYSTEM. ONE SOURCE OF TRUTH. NO EXCEPTIONS.**

**Last Verification:** October 1, 2025
**Next Check:** Before ANY discovery system modifications
**Lock Status:** 🔒 ACTIVE AND ENFORCED
