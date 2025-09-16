#!/usr/bin/env python3
"""
Configuration constants for Discovery Pipeline v2.0.1
"""
import time
from typing import Dict, Any

# Schema version
SCHEMA_VERSION = "2.0.1"

# TTL constants (in seconds)
TTL_SNAPSHOT = 120  # 1-2 min
TTL_SHORT_BORROW = 3600  # 1 hour
TTL_OPTIONS = 600  # 5-10 min
TTL_SENTIMENT = 300  # 1-5 min
TTL_REFERENCE = 86400  # nightly

# Exclude types for Gate A hygiene
EXCLUDE_TYPES = ("etf", "etn", "fund", "reit", "cef", "adr")

# Lane thresholds
LANE_A_PRICE_THRESH = 10.0
LANE_A_FLOAT_THRESH = 50_000_000

# Scoring caps
RVOL_CAP = 6.0
PERCENT_CHANGE_CAP = 20.0
ATR_PERCENT_CAP = 12.0

# RVOL sustained thresholds
RVOL_SUST_THRESH = 3.0
RVOL_SUST_MIN_MINUTES = 30

# Options rule thresholds
MIN_IV_PERCENTILE = 80
MIN_CALL_PUT_OI_RATIO = 2.0

# Float/Short rule thresholds
LARGE_FLOAT_THRESH = 150_000_000
MIN_SHORT_INTEREST_PCT = 20
MIN_BORROW_FEE_PCT = 20
MIN_UTILIZATION_PCT = 85

# P/E threshold
MAX_PE_RATIO = 50

def is_fresh(timestamp: float, ttl: float) -> bool:
    """Check if data is fresh based on TTL"""
    return (time.time() - timestamp) < ttl

# Drop reason enums
class DropReason:
    DROP_NON_COMMON = "DROP_NON_COMMON"
    DROP_RVOL_NOT_SUSTAINED = "DROP_RVOL_NOT_SUSTAINED"
    DROP_BELOW_VWAP = "DROP_BELOW_VWAP"
    DROP_EMA_FALSE = "DROP_EMA_FALSE"
    DROP_OPTIONS_RULE = "DROP_OPTIONS_RULE"
    DROP_FLOAT_SHORT_RULE = "DROP_FLOAT_SHORT_RULE"
    DROP_TTL_STALE = "DROP_TTL_STALE"
    DROP_PE_HIGH = "DROP_PE_HIGH"