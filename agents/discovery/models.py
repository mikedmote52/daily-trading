#!/usr/bin/env python3
"""
Pydantic models for Discovery Pipeline v2.0.1
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, validator
from datetime import datetime
from config import SCHEMA_VERSION

class StockCandidate(BaseModel):
    """Individual stock candidate output model"""
    symbol: str
    price: float
    market_cap: float
    lane: str  # "A" or "B"
    status: str  # "TRADE_READY", "WATCHLIST", "FILTERED_OUT"
    
    # Scoring
    explosive_score: int
    bucket_scores: Dict[str, int]
    
    # Key metrics
    rvol_sust: float
    percent_change: float
    volume_surge: float
    short_interest_pct: float
    iv_percentile: Optional[float]
    call_put_oi_ratio: Optional[float]
    
    # Technical indicators
    rsi: Optional[float]
    vwap: Optional[float]
    ema9: Optional[float]
    ema20: Optional[float]
    
    # Metadata
    sector: Optional[str]
    exchange: Optional[str]
    drop_reason: Optional[str]
    warnings: List[str] = []
    sustained_30m: bool = False
    
    @validator('explosive_score')
    def score_range(cls, v):
        assert 0 <= v <= 100, 'Score must be 0-100'
        return v
    
    @validator('bucket_scores')
    def bucket_score_range(cls, v):
        for bucket, score in v.items():
            assert 0 <= score <= 100, f'Bucket {bucket} score must be 0-100'
        return v

class GateMetrics(BaseModel):
    """Gate processing metrics"""
    input_count: int
    output_count: int
    processing_time_seconds: float
    drop_reasons: Dict[str, int] = {}

class DiscoveryResult(BaseModel):
    """Complete discovery pipeline result"""
    schema_version: str = SCHEMA_VERSION
    timestamp: datetime
    processing_time_seconds: float
    
    # Results
    candidates: List[StockCandidate]
    trade_ready_count: int
    watchlist_count: int
    
    # Metrics
    gate_a_metrics: GateMetrics
    gate_b_metrics: GateMetrics  
    gate_c_metrics: GateMetrics
    total_drops: Dict[str, int]
    
    # Data freshness
    data_freshness: Dict[str, bool]
    
    @validator('candidates')
    def no_null_fields(cls, v):
        for candidate in v:
            # Ensure no None values in required fields
            assert candidate.symbol is not None
            assert candidate.price is not None
            assert candidate.explosive_score is not None
        return v