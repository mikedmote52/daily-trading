# CRITICAL SYSTEM RULES - MUST FOLLOW

## 🚨 SINGLE DISCOVERY SYSTEM RULE

### THE ONE AND ONLY DISCOVERY SYSTEM:
**Location:** `/Users/michaelmote/Desktop/Daily-Trading/agents/discovery/universal_discovery.py`

### STRICT RULES:
1. **NEVER CREATE NEW DISCOVERY FILES** - Only edit `universal_discovery.py`
2. **NEVER CREATE "FIXED" VERSIONS** - No `fixed_universal_discovery.py`, `enhanced_discovery.py`, etc.
3. **NEVER CREATE BACKUP/ALTERNATE SYSTEMS** - One system, one source of truth
4. **ALWAYS CHECK BEFORE CREATING** - Run `find . -name "*discovery*.py"` before any work
5. **EDIT, DON'T RECREATE** - Use Edit/MultiEdit tools on existing file only

### FORBIDDEN ACTIONS:
- ❌ Creating `fixed_universal_discovery.py`
- ❌ Creating `improved_discovery.py`  
- ❌ Creating `enhanced_discovery_engine.py`
- ❌ Creating any new discovery system file
- ❌ Creating "test" discovery systems that become permanent

### REQUIRED ACTIONS:
- ✅ ALWAYS edit `/agents/discovery/universal_discovery.py` directly
- ✅ ALWAYS remove any duplicate discovery files immediately
- ✅ ALWAYS verify single system with: `ls -la | grep discovery`

## ENFORCEMENT CHECKLIST:
Before ANY discovery system work:
1. [ ] Confirmed location: `/agents/discovery/universal_discovery.py`
2. [ ] Checked for duplicates: `find . -name "*discovery*.py"`
3. [ ] Using Edit/MultiEdit on existing file ONLY
4. [ ] NOT creating any new files

## CONSEQUENCES OF VIOLATIONS:
- System failures due to competing discovery systems
- Inconsistent results from different implementations
- Wasted computational resources
- Confusion about which system is authoritative

## RECOVERY PROTOCOL:
If duplicates are found:
1. Immediately identify all discovery files
2. Keep ONLY `universal_discovery.py`
3. Delete ALL other discovery implementations
4. Verify single system remains

---
**REMEMBER: ONE DISCOVERY SYSTEM. ONE SOURCE OF TRUTH. NO EXCEPTIONS.**