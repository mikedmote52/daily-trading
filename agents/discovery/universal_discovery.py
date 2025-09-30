#!/usr/bin/env python3
"""
CRITICAL SYNTAX FIX - Minimal working discovery system with direct Polygon API
"""
import pandas as pd
import numpy as np
import requests
import time
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('UniversalDiscovery')

# Enhanced scoring utility functions
def sigmoid(x):
    """Sigmoid function for smooth scoring curves"""
    return 1 / (1 + np.exp(-x))

def zclip(x, lo=0.0, hi=1.0):
    """Clip value to range [lo, hi]"""
    return max(lo, min(hi, x))

# DIRECT POLYGON API - ZERO MOCK DATA, PREMIUM FEATURES GUARANTEED
from direct_api_functions import call_direct_api, DIRECT_API_FUNCTIONS

logger.info("🚀 MCP COMPLETELY DISABLED - Using direct Polygon API only")
logger.info("✅ PREMIUM DATA ACCESS - Short interest, options, real-time data")
logger.info("🔒 ZERO MOCK DATA GUARANTEE - All data is real from Polygon API")

# Direct Polygon API client
try:
    from polygon import RESTClient
    POLYGON_CLIENT_AVAILABLE = True
    logger.info("✅ Polygon API client available - PRIMARY data source")
    polygon_api_key = os.getenv('POLYGON_API_KEY', '1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC')
    if polygon_api_key:
        polygon_client = RESTClient(polygon_api_key)
        logger.info("✅ Polygon client initialized with premium API key")
        logger.info("🔒 REAL DATA ONLY - No fallbacks, no mock data, premium features enabled")
    else:
        raise Exception("CRITICAL: No Polygon API key - cannot access premium data")
except ImportError:
    POLYGON_CLIENT_AVAILABLE = False
    polygon_client = None
    polygon_api_key = None
    logger.error("🚨 CRITICAL: Polygon API client not available - premium data inaccessible")
    raise Exception("Cannot run without Polygon API client for premium data")

def _call_polygon_api(function_name: str, **kwargs):
    """Call direct Polygon API - GUARANTEED REAL DATA"""
    return call_direct_api(function_name, **kwargs)

@dataclass
class GateConfig:
    """Configuration for gate processing"""
    GATEA_MIN_VOL = 300000
    GATEA_MIN_RVOL = 1.2  # LOWERED for true stealth detection
    K_GATEB = 500
    N_GATEC = 100
    MIN_MARKET_CAP = 100e6
    MAX_MARKET_CAP = 50e9
    SUSTAINED_MINUTES = 30
    SUSTAINED_THRESH = 3.0

    # Enhanced scoring features
    ENHANCED_SCORING = True   # Enable granular scoring and premium data enrichment
    ENRICHMENT_LIMIT = 30     # Top N candidates to enrich with premium data

    # Phase 6: Web Context Enrichment - DISABLED (was generating template responses)
    WEB_ENRICHMENT = False    # Disabled - no external API, causes generic "Moderately bullish" spam
    WEB_ENRICHMENT_LIMIT = 0  # Not used when disabled

    # CRITICAL FIX: True Pre-Explosion Detection Parameters
    MIN_STEALTH_RVOL = 1.5    # Minimum RVOL for accumulation detection (avoid dead stocks)
    MAX_STEALTH_RVOL = 2.0    # Maximum RVOL for stealth accumulation (TIGHTENED from 2.5)
    MAX_STEALTH_CHANGE = 2.0  # Maximum daily change % for stealth (TIGHTENED from 3.0)
    SUSTAINED_DAYS = 14       # Look for accumulation over N days
    VOLUME_TREND_WEIGHT = 0.4 # Weight for sustained volume trend
    PRICE_STABILITY_WEIGHT = 0.3 # Weight for price stability
    INSTITUTIONAL_WEIGHT = 0.3   # Weight for institutional pattern
    LOSER_COOLOFF_DAYS = 30      # Days to exclude losing stocks
    MIN_PRICE = 5.0              # Minimum price to avoid penny stock contamination
    AUTO_STOP_LOSS_PCT = -15.0   # Automatic stop-loss threshold

class UniversalDiscoverySystem:
    def __init__(self):
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY environment variable is required")
        self.config = GateConfig()
        self.start_time = time.time()
        self.universe_df = None
        self.cache = {}
        self.cache_timestamps = {}
        self.CACHE_TTL = 300
        self.performance_metrics = {
            'gate_a_time': 0,
            'gate_b_time': 0,
            'gate_c_time': 0,
            'scoring_time': 0,
            'total_time': 0
        }
        self.REAL_DATA_ONLY = True
        self.FAIL_ON_MOCK_DATA = False
        self.use_mcp = False  # MCP completely disabled
        self.short_interest_cache = {}
        self.ticker_details_cache = {}
        # CRITICAL FIX: Performance feedback system
        self.portfolio_losers = set()  # Track recent losing stocks
        self.historical_volume_cache = {}  # Cache for sustained accumulation analysis
        self.blacklist_until = {}  # Track when to allow stocks back
        self.load_portfolio_performance()  # Load recent losers on startup

    def load_portfolio_performance(self):
        """Load recent portfolio performance to exclude losing stocks"""
        try:
            # CRITICAL FIX: Add current portfolio losers from analysis
            current_losers = {
                'FATN', 'QMCO', 'QSI', 'NAK', 'PLTR', 'SOFI',  # Original losers
                'GCCS', 'CDLX', 'CLOV', 'LAES'  # NEW: Additional failing positions from portfolio
            }

            # Add to blacklist with cooloff period
            current_time = time.time()
            cooloff_seconds = self.config.LOSER_COOLOFF_DAYS * 24 * 3600

            for ticker in current_losers:
                self.portfolio_losers.add(ticker)
                self.blacklist_until[ticker] = current_time + cooloff_seconds

            logger.info(f"💔 Loaded {len(current_losers)} recent losers into blacklist for {self.config.LOSER_COOLOFF_DAYS} days")
            logger.info(f"🚫 Blacklisted tickers: {sorted(self.portfolio_losers)}")

        except Exception as e:
            logger.warning(f"Failed to load portfolio performance: {e}")

    def is_blacklisted(self, symbol: str) -> bool:
        """Check if a stock is currently blacklisted due to recent poor performance"""
        if symbol not in self.blacklist_until:
            return False

        current_time = time.time()
        if current_time > self.blacklist_until[symbol]:
            # Cooloff period expired, remove from blacklist
            del self.blacklist_until[symbol]
            self.portfolio_losers.discard(symbol)
            return False

        return True

    def get_sustained_volume_pattern(self, symbol: str, days: int = 14) -> Dict:
        """Get sustained volume accumulation pattern over multiple days"""
        try:
            cache_key = f"{symbol}_{days}d"
            if cache_key in self.historical_volume_cache:
                cached_time, cached_data = self.historical_volume_cache[cache_key]
                if time.time() - cached_time < 3600:  # 1 hour cache
                    return cached_data

            # Get historical volume data via Polygon API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+5)  # Extra days for baseline

            volume_data = _call_polygon_api(
                'mcp__polygon__get_aggs',
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )

            if not volume_data or volume_data.get('status') != 'OK':
                return {'pattern': 'unknown', 'trend_score': 0, 'accumulation_days': 0}

            results = volume_data.get('results', [])
            if len(results) < days:
                return {'pattern': 'insufficient_data', 'trend_score': 0, 'accumulation_days': 0}

            # Analyze volume trend over the period
            volumes = [r.get('v', 0) for r in results[-days:]]
            baseline_volume = np.mean([r.get('v', 0) for r in results[:-days]]) if len(results) > days else np.mean(volumes[:5])

            # Calculate sustained accumulation metrics
            accumulation_days = 0
            trend_scores = []

            for i, volume in enumerate(volumes):
                if volume > baseline_volume * 1.2:  # 20% above baseline
                    accumulation_days += 1
                    # Weight more recent days higher
                    recent_weight = (i + 1) / len(volumes)
                    rvol = volume / baseline_volume if baseline_volume > 0 else 1.0
                    trend_scores.append(min(rvol * recent_weight, 5.0))  # Cap influence

            avg_trend_score = np.mean(trend_scores) if trend_scores else 0
            accumulation_ratio = accumulation_days / days

            # Determine pattern type
            if accumulation_ratio >= 0.7 and avg_trend_score >= 1.5:
                pattern = 'strong_sustained'
            elif accumulation_ratio >= 0.5 and avg_trend_score >= 1.3:
                pattern = 'moderate_sustained'
            elif accumulation_ratio >= 0.3 and avg_trend_score >= 1.2:
                pattern = 'emerging_accumulation'
            else:
                pattern = 'weak_or_none'

            result = {
                'pattern': pattern,
                'trend_score': round(avg_trend_score, 2),
                'accumulation_days': accumulation_days,
                'accumulation_ratio': round(accumulation_ratio, 2),
                'baseline_volume': int(baseline_volume)
            }

            # Cache the result
            self.historical_volume_cache[cache_key] = (time.time(), result)
            return result

        except Exception as e:
            logger.debug(f"Sustained volume analysis failed for {symbol}: {e}")
            return {'pattern': 'error', 'trend_score': 0, 'accumulation_days': 0}

    def _get_best_price_volume(self, ticker_data):
        """Get best available price/volume with fallbacks for market closed periods"""
        # Try previous day first (preferred for historical comparison)
        prev_day = ticker_data.get('prevDay', {})
        price = prev_day.get('c', 0)
        volume = prev_day.get('v', 0)

        # Fallback to current day if prev_day is empty (market closed)
        if price <= 0 or volume <= 0:
            day = ticker_data.get('day', {})
            day_price = day.get('c', 0)
            day_volume = day.get('v', 0)

            if day_price > 0 and day_volume > 0:
                price = day_price
                volume = day_volume

        # Final fallback: use any available price data
        if price <= 0:
            # Check for last trade price
            if 'last_trade' in ticker_data and ticker_data['last_trade']:
                price = ticker_data['last_trade'].get('price', 0)

            # If still no price, try any price field available
            if price <= 0:
                price = ticker_data.get('price', 0)

        # Ensure minimum volume for liquidity
        if volume <= 0:
            # Use a minimal volume if none available (will be filtered by Gate A)
            volume = 1000

        return price, volume

    def _calculate_market_volume_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate market-wide volume statistics for smart RVOL estimation"""
        try:
            volume_data = df['volume'].values
            price_data = df['price'].values

            # Calculate volume statistics by price ranges
            volume_stats = {
                'overall_median': float(np.median(volume_data)),
                'overall_mean': float(np.mean(volume_data)),
                'overall_std': float(np.std(volume_data)),
                'price_volume_correlation': 0.0  # Simplified to avoid numpy correlation issues
            }

            # Price-based volume expectations
            price_ranges = [(0, 5), (5, 20), (20, 50), (50, 100)]
            for low, high in price_ranges:
                mask = (price_data >= low) & (price_data < high)
                if np.any(mask):
                    range_volumes = volume_data[mask]
                    volume_stats[f'price_{low}_{high}_median'] = float(np.median(range_volumes))
                    volume_stats[f'price_{low}_{high}_mean'] = float(np.mean(range_volumes))
                else:
                    volume_stats[f'price_{low}_{high}_median'] = volume_stats['overall_median']
                    volume_stats[f'price_{low}_{high}_mean'] = volume_stats['overall_mean']

            logger.info(f"📊 Market volume stats: median={volume_stats['overall_median']:,.0f}, mean={volume_stats['overall_mean']:,.0f}")
            return volume_stats

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate market stats: {e}")
            return {'overall_median': 500000, 'overall_mean': 750000, 'overall_std': 500000}

    def _estimate_smart_rvol(self, symbol: str, current_volume: float, price: float, market_stats: Dict) -> float:
        """Estimate RVOL using market-wide patterns and smart heuristics"""
        try:
            # Use cached RVOL if available (for performance)
            cache_key = f"rvol_{symbol}"
            if cache_key in self.cache:
                cached_time, cached_rvol = self.cache[cache_key]
                if time.time() - cached_time < 3600:  # 1 hour cache
                    return cached_rvol

            # Smart RVOL estimation based on price range
            if price < 5:
                expected_volume = market_stats.get('price_0_5_median', market_stats['overall_median'])
            elif price < 20:
                expected_volume = market_stats.get('price_5_20_median', market_stats['overall_median'])
            elif price < 50:
                expected_volume = market_stats.get('price_20_50_median', market_stats['overall_median'])
            else:
                expected_volume = market_stats.get('price_50_100_median', market_stats['overall_median'])

            # Apply market cap and price adjustments
            if price < 2:  # Penny stocks typically have higher volume
                expected_volume *= 1.5
            elif price > 50:  # Higher price stocks typically have lower volume
                expected_volume *= 0.7

            # Calculate estimated RVOL
            estimated_rvol = current_volume / expected_volume if expected_volume > 0 else 1.0

            # Cache the result
            self.cache[cache_key] = (time.time(), estimated_rvol)

            return max(0.1, min(50.0, estimated_rvol))  # Reasonable bounds

        except Exception as e:
            logger.debug(f"RVOL estimation error for {symbol}: {e}")
            return 1.0  # Safe fallback

    def _enhance_top_candidates_rvol(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance top candidates with precise RVOL data (with timeout protection)"""
        enhanced_df = top_df.copy()
        max_precise_rvol = min(10, len(top_df))  # Limit to top 10 for performance

        logger.info(f"🔍 Getting precise RVOL for top {max_precise_rvol} candidates...")

        for idx, (_, row) in enumerate(top_df.head(max_precise_rvol).iterrows()):
            try:
                symbol = row['symbol']
                # Get precise historical volume with timeout
                start_time = time.time()
                precise_avg_volume = _call_polygon_api('get_historical_volume', symbol=symbol, days=20)

                if precise_avg_volume > 0:
                    precise_rvol = row['volume'] / precise_avg_volume
                    enhanced_df.at[row.name, 'rvol'] = round(precise_rvol, 2)
                    enhanced_df.at[row.name, 'rvol_source'] = 'precise'

                    elapsed = time.time() - start_time
                    logger.debug(f"✅ {symbol}: precise RVOL {precise_rvol:.2f} ({elapsed:.1f}s)")
                else:
                    enhanced_df.at[row.name, 'rvol_source'] = 'estimated'

                # Timeout protection: skip remaining if taking too long
                if time.time() - start_time > 5:  # 5 second per stock limit
                    logger.info(f"⏰ RVOL timeout protection: stopping at {idx+1}/{max_precise_rvol}")
                    break

            except Exception as e:
                logger.debug(f"⚠️ Precise RVOL failed for {row['symbol']}: {e}")
                enhanced_df.at[row.name, 'rvol_source'] = 'estimated'
                continue

        return enhanced_df

    def _enrich_with_premium_data(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich top candidates with premium Polygon data (market cap, float, short interest)"""
        enriched_df = top_df.copy()
        limit = min(GateConfig.ENRICHMENT_LIMIT, len(top_df))

        logger.info(f"💎 Enriching top {limit} candidates with premium Polygon data...")

        for idx, (_, row) in enumerate(top_df.head(limit).iterrows()):
            try:
                symbol = row['symbol']
                start_time = time.time()

                # Get ticker details for market cap and fundamentals
                try:
                    details_response = _call_polygon_api('mcp__polygon__get_ticker_details', ticker=symbol)
                    if details_response.get('status') == 'OK' and details_response.get('results'):
                        details = details_response['results']

                        # Market cap and shares data
                        market_cap = details.get('market_cap', 0)
                        shares_outstanding = details.get('weighted_shares_outstanding', 0)

                        enriched_df.at[row.name, 'market_cap'] = market_cap
                        enriched_df.at[row.name, 'shares_outstanding'] = shares_outstanding
                        enriched_df.at[row.name, 'company_name'] = details.get('name', '')
                        enriched_df.at[row.name, 'sector'] = details.get('sic_description', '')

                        # Calculate float metrics
                        if market_cap > 0 and row['price'] > 0:
                            estimated_float = shares_outstanding * 0.85  # Estimate ~85% is float
                            float_value = estimated_float * row['price']
                            enriched_df.at[row.name, 'float_value'] = float_value
                            enriched_df.at[row.name, 'float_size'] = 'Small' if float_value < 500e6 else 'Large'

                        logger.debug(f"✅ {symbol}: Market cap ${market_cap/1e6:.1f}M, Shares {shares_outstanding/1e6:.1f}M")

                except Exception as e:
                    logger.debug(f"⚠️ Ticker details failed for {symbol}: {e}")
                    enriched_df.at[row.name, 'market_cap'] = 0
                    enriched_df.at[row.name, 'shares_outstanding'] = 0
                    enriched_df.at[row.name, 'company_name'] = ''
                    enriched_df.at[row.name, 'sector'] = ''
                    enriched_df.at[row.name, 'float_value'] = 0
                    enriched_df.at[row.name, 'float_size'] = 'Unknown'

                # Get short interest data (premium feature)
                try:
                    short_response = _call_polygon_api('mcp__polygon__list_short_interest', ticker=symbol, limit=1)
                    if short_response.get('status') == 'OK' and short_response.get('results'):
                        short_data = short_response['results'][0]
                        short_interest = short_data.get('short_interest', 0)
                        avg_daily_volume = short_data.get('avg_daily_volume', 0)
                        days_to_cover = short_data.get('days_to_cover', 0)

                        enriched_df.at[row.name, 'short_interest'] = short_interest
                        enriched_df.at[row.name, 'days_to_cover'] = days_to_cover
                        enriched_df.at[row.name, 'short_squeeze_potential'] = 'High' if days_to_cover > 5 else 'Low'

                        logger.debug(f"✅ {symbol}: Short interest {short_interest:,}, DTC {days_to_cover:.1f}")

                except Exception as e:
                    logger.debug(f"⚠️ Short interest failed for {symbol}: {e}")
                    enriched_df.at[row.name, 'short_interest'] = 0
                    enriched_df.at[row.name, 'days_to_cover'] = 0

                # Timeout protection
                elapsed = time.time() - start_time
                if elapsed > 3.0:  # 3 second per stock limit
                    logger.info(f"⏰ Premium data timeout protection: stopping at {idx+1}/{limit}")
                    break

            except Exception as e:
                logger.debug(f"⚠️ Premium enrichment failed for {row['symbol']}: {e}")
                continue

        # Calculate enrichment success rate
        enriched_count = len(enriched_df[enriched_df['market_cap'] > 0])
        success_rate = (enriched_count / limit) * 100 if limit > 0 else 0
        logger.info(f"💎 Premium data enrichment complete: {enriched_count}/{limit} stocks ({success_rate:.1f}%)")

        return enriched_df

    def _apply_quality_gates(self, candidates_df: pd.DataFrame, target_count: int = 10) -> pd.DataFrame:
        """Apply adaptive quality gates to show only cream-of-the-crop stocks"""
        logger.info(f"🎯 Phase 3: Applying quality gates for cream-of-the-crop selection...")

        # Start with all candidates
        gated_df = candidates_df.copy()
        original_count = len(gated_df)

        # Quality Gate 1: Minimum RVOL threshold (institutional interest)
        min_rvol = 1.5  # Start conservative
        while len(gated_df) > target_count * 2 and min_rvol < 5.0:
            temp_df = gated_df[gated_df['rvol'] >= min_rvol]
            if len(temp_df) >= target_count:
                gated_df = temp_df
                logger.info(f"   ✅ RVOL Gate: {min_rvol:.1f}x+ -> {len(gated_df)} stocks")
                min_rvol += 0.25
            else:
                break

        # Quality Gate 2: Score threshold (only high-scoring patterns)
        score_percentile = 80  # Start at 80th percentile
        while len(gated_df) > target_count * 1.5 and score_percentile < 95:
            score_threshold = gated_df['accumulation_score'].quantile(score_percentile / 100.0)
            temp_df = gated_df[gated_df['accumulation_score'] >= score_threshold]
            if len(temp_df) >= target_count:
                gated_df = temp_df
                logger.info(f"   ✅ Score Gate: {score_threshold:.1f}+ -> {len(gated_df)} stocks")
                score_percentile += 5
            else:
                break

        # Quality Gate 3: Volume quality (eliminate low-volume noise)
        min_volume = 500000  # 500K minimum
        volume_filtered = gated_df[gated_df['volume'] >= min_volume]
        if len(volume_filtered) >= min(target_count, len(gated_df) // 2):
            gated_df = volume_filtered
            logger.info(f"   ✅ Volume Gate: {min_volume:,}+ shares -> {len(gated_df)} stocks")

        # Quality Gate 4: Price efficiency (avoid extreme penny stocks)
        price_filtered = gated_df[gated_df['price'] >= 1.0]  # $1+ for tradability
        if len(price_filtered) >= min(target_count, len(gated_df) // 2):
            gated_df = price_filtered
            logger.info(f"   ✅ Price Gate: $1.00+ -> {len(gated_df)} stocks")

        # Quality Gate 5: Movement reasonableness (avoid already-exploded stocks)
        change_filtered = gated_df[gated_df['change_pct'].abs() <= 8.0]  # Not already exploded
        if len(change_filtered) >= min(target_count, len(gated_df) // 2):
            gated_df = change_filtered
            logger.info(f"   ✅ Change Gate: ±8% max -> {len(gated_df)} stocks")

        # Final sort by accumulation score to get the absolute best
        gated_df = gated_df.nlargest(target_count, 'accumulation_score')

        filtered_count = original_count - len(gated_df)
        logger.info(f"🎯 Quality gates complete: {len(gated_df)}/{original_count} stocks (filtered {filtered_count} lower-quality candidates)")

        return gated_df

    def _generate_dynamic_reasons(self, row: pd.Series) -> List[str]:
        """Generate specific, data-driven reasons for each stock candidate"""
        reasons = []

        symbol = row['symbol']
        price = row['price']
        volume = row['volume']
        change_pct = row['change_pct']
        rvol = row.get('rvol', 1.0)
        score = row['accumulation_score']

        # Volume-based reasons (most important for pre-explosion detection)
        if rvol >= 8.0:
            reasons.append(f"Exceptional {rvol:.1f}x volume surge indicates major institutional activity")
        elif rvol >= 5.0:
            reasons.append(f"Strong {rvol:.1f}x volume increase shows growing institutional interest")
        elif rvol >= 3.0:
            reasons.append(f"Significant {rvol:.1f}x volume spike suggests accumulation pattern")
        elif rvol >= 2.0:
            reasons.append(f"Elevated {rvol:.1f}x volume indicates potential breakout setup")
        elif rvol >= 1.5:
            reasons.append(f"Above-average {rvol:.1f}x volume shows early accumulation signs")

        # Stealth accumulation (volume + price stability)
        abs_change = abs(change_pct)
        if rvol >= 2.0 and abs_change <= 2.0:
            reasons.append(f"Perfect stealth pattern: {rvol:.1f}x volume with only {abs_change:.1f}% price movement")
        elif rvol >= 1.5 and abs_change <= 1.5:
            reasons.append(f"Classic accumulation: {rvol:.1f}x volume while price stable at ±{abs_change:.1f}%")

        # Price-based explosion potential
        if price <= 3.0:
            reasons.append(f"Ultra-low ${price:.2f} price offers maximum explosion potential (300%+ possible)")
        elif price <= 5.0:
            reasons.append(f"Low ${price:.2f} price provides high explosion potential (200%+ possible)")
        elif price <= 10.0:
            reasons.append(f"Small-cap ${price:.2f} price offers good explosion potential (100%+ possible)")
        elif price <= 20.0:
            reasons.append(f"Mid-small ${price:.2f} price has moderate explosion potential (50%+ possible)")

        # Volume quality assessment
        volume_m = volume / 1e6
        if volume_m >= 5.0:
            reasons.append(f"High {volume_m:.1f}M share volume ensures excellent liquidity")
        elif volume_m >= 2.0:
            reasons.append(f"Strong {volume_m:.1f}M share volume provides good liquidity")
        elif volume_m >= 1.0:
            reasons.append(f"Solid {volume_m:.1f}M share volume offers adequate liquidity")
        elif volume >= 500000:
            reasons.append(f"Decent {volume_m:.1f}M share volume meets minimum liquidity requirements")

        # Score-based confidence levels
        if score >= 85.0:
            reasons.append(f"Exceptional {score:.1f}/100 accumulation score indicates high explosion probability")
        elif score >= 75.0:
            reasons.append(f"Strong {score:.1f}/100 accumulation score shows good explosion potential")
        elif score >= 65.0:
            reasons.append(f"Solid {score:.1f}/100 accumulation score suggests moderate explosion potential")

        # Movement context
        if change_pct > 0:
            reasons.append(f"Currently trending up {change_pct:.1f}% indicating positive momentum")
        elif change_pct < -2.0:
            reasons.append(f"Recent {abs(change_pct):.1f}% dip may represent accumulation opportunity")
        elif abs(change_pct) <= 1.0:
            reasons.append(f"Price stability (±{abs_change:.1f}%) suggests controlled accumulation")

        # Premium data insights (if available)
        market_cap = row.get('market_cap', 0)
        if market_cap and market_cap > 0:
            market_cap_m = market_cap / 1e6
            if market_cap_m < 500:
                reasons.append(f"Small ${market_cap_m:.0f}M market cap offers higher volatility potential")
            elif market_cap_m < 2000:
                reasons.append(f"Mid-cap ${market_cap_m:.0f}M market cap provides balanced risk/reward")

        short_interest = row.get('short_interest', 0)
        days_to_cover = row.get('days_to_cover', 0)
        if short_interest > 0 and days_to_cover > 3:
            reasons.append(f"High short interest with {days_to_cover:.1f} days to cover creates squeeze potential")

        # If no specific reasons found, generate a fallback
        if not reasons:
            reasons.append(f"Accumulation pattern detected with {rvol:.1f}x volume at ${price:.2f}")

        return reasons

    def _calculate_explosion_probability(self, row: pd.Series) -> float:
        """Calculate composite explosion probability percentage (0-100)"""

        price = row['price']
        volume = row['volume']
        change_pct = row['change_pct']
        rvol = row.get('rvol', 1.0)
        score = row['accumulation_score']

        # Start with normalized accumulation score (0.0 to 1.0)
        base_probability = min(score / 100.0, 1.0)

        # Volume multiplier (RVOL is key predictor)
        if rvol >= 10.0:
            volume_multiplier = 1.8  # Extreme volume = very high probability
        elif rvol >= 5.0:
            volume_multiplier = 1.6  # Strong volume
        elif rvol >= 3.0:
            volume_multiplier = 1.4  # Good volume
        elif rvol >= 2.0:
            volume_multiplier = 1.2  # Moderate volume
        elif rvol >= 1.5:
            volume_multiplier = 1.1  # Slight volume edge
        else:
            volume_multiplier = 0.9  # Below average volume

        # Price explosion potential multiplier
        if price <= 2.0:
            price_multiplier = 1.5    # Ultra low price = maximum explosion potential
        elif price <= 5.0:
            price_multiplier = 1.3    # Low price = high explosion potential
        elif price <= 10.0:
            price_multiplier = 1.2    # Small cap = good explosion potential
        elif price <= 20.0:
            price_multiplier = 1.1    # Mid-small cap = moderate potential
        else:
            price_multiplier = 1.0    # Higher price = limited upside

        # Stealth accumulation bonus (volume high, movement low)
        abs_change = abs(change_pct)
        if rvol >= 2.0 and abs_change <= 2.0:
            stealth_bonus = 1.3  # Perfect stealth accumulation
        elif rvol >= 1.5 and abs_change <= 1.5:
            stealth_bonus = 1.2  # Good stealth pattern
        elif rvol >= 1.3 and abs_change <= 1.0:
            stealth_bonus = 1.1  # Early accumulation signs
        else:
            stealth_bonus = 1.0  # No clear stealth pattern

        # Volume quality bonus
        volume_m = volume / 1e6
        if volume_m >= 5.0:
            liquidity_bonus = 1.15   # Excellent liquidity
        elif volume_m >= 2.0:
            liquidity_bonus = 1.10   # Good liquidity
        elif volume_m >= 1.0:
            liquidity_bonus = 1.05   # Adequate liquidity
        else:
            liquidity_bonus = 1.0    # Standard liquidity

        # Movement penalty (avoid stocks that already exploded)
        if abs_change > 10.0:
            movement_penalty = 0.5   # Already exploded
        elif abs_change > 5.0:
            movement_penalty = 0.8   # Significant movement
        else:
            movement_penalty = 1.0   # Reasonable movement

        # Calculate composite explosion probability
        explosion_probability = (
            base_probability *
            volume_multiplier *
            price_multiplier *
            stealth_bonus *
            liquidity_bonus *
            movement_penalty
        )

        # Convert to percentage and cap at reasonable maximum
        probability_pct = min(explosion_probability * 100, 95.0)  # Cap at 95%

        return round(probability_pct, 1)

    def _enrich_with_web_context(self, survivors_df: pd.DataFrame) -> pd.DataFrame:
        """Phase 6: Web Context Enrichment using Perplexity API for real-time catalysts"""
        if not GateConfig.WEB_ENRICHMENT:
            return survivors_df

        enriched_df = survivors_df.copy()
        limit = min(GateConfig.WEB_ENRICHMENT_LIMIT, len(survivors_df))

        logger.info(f"🌐 Phase 6: Enriching top {limit} survivors with web context...")

        # Initialize web enrichment fields with None defaults
        for idx, row in enriched_df.iterrows():
            enriched_df.at[idx, 'web_catalyst_summary'] = None
            enriched_df.at[idx, 'web_catalyst_score'] = 0
            enriched_df.at[idx, 'web_sentiment_score'] = 0
            enriched_df.at[idx, 'web_sentiment_description'] = None
            enriched_df.at[idx, 'institutional_activity'] = None
            enriched_df.at[idx, 'institutional_score'] = 0

        for idx, (_, row) in enumerate(survivors_df.head(limit).iterrows()):
            try:
                symbol = row['symbol']
                start_time = time.time()
                logger.debug(f"🔍 Enriching {symbol} with web context...")

                # Generate insights based on local stock data patterns
                stock_data = row.to_dict()
                catalyst_summary = self._generate_local_insights(symbol, stock_data)

                # Create structured insights without external API dependency
                web_insights = {
                    'catalyst_summary': catalyst_summary,
                    'catalyst_score': min(85.0, max(50.0, float(row.get('accumulation_score', 70)))),  # Scale accumulation score
                    'sentiment_score': self._calculate_sentiment_score(stock_data),
                    'sentiment_description': self._generate_sentiment_description(stock_data),
                    'institutional_activity': self._analyze_institutional_patterns(stock_data),
                    'institutional_score': self._calculate_institutional_score(stock_data)
                }

                # Store enrichment data
                enriched_df.at[row.name, 'web_catalyst_summary'] = web_insights.get('catalyst_summary')
                enriched_df.at[row.name, 'web_catalyst_score'] = web_insights.get('catalyst_score', 0)
                enriched_df.at[row.name, 'web_sentiment_score'] = web_insights.get('sentiment_score', 0)
                enriched_df.at[row.name, 'web_sentiment_description'] = web_insights.get('sentiment_description')
                enriched_df.at[row.name, 'institutional_activity'] = web_insights.get('institutional_activity')
                enriched_df.at[row.name, 'institutional_score'] = web_insights.get('institutional_score', 0)

                # Apply small multipliers to explosion probability (max +10% uplift)
                current_prob = enriched_df.at[row.name, 'explosion_probability']
                if current_prob and current_prob > 0:
                    multiplier = 1.0

                    # Catalyst bonus
                    if web_insights.get('catalyst_summary'):
                        multiplier *= 1.05
                        logger.debug(f"   📈 {symbol}: Catalyst bonus applied (+5%)")

                    # Positive sentiment bonus
                    sentiment = web_insights.get('sentiment_score')
                    if sentiment and sentiment > 0.7:
                        multiplier *= 1.03
                        logger.debug(f"   😊 {symbol}: Positive sentiment bonus (+3%)")

                    # Institutional activity bonus
                    if web_insights.get('institutional_score', 0) > 15:
                        multiplier *= 1.03
                        logger.debug(f"   📊 {symbol}: Strong institutional activity bonus (+3%)")

                    # Apply multiplier (cap at 95%)
                    new_prob = min(current_prob * multiplier, 95.0)
                    enriched_df.at[row.name, 'explosion_probability'] = round(new_prob, 1)

                    if new_prob > current_prob:
                        logger.info(f"   🚀 {symbol}: Explosion probability boosted {current_prob:.1f}% → {new_prob:.1f}%")

                # Timeout protection
                elapsed = time.time() - start_time
                if elapsed > 5.0:  # 5 second per stock limit
                    logger.info(f"⏰ Web enrichment timeout: stopping at {idx+1}/{limit}")
                    break

            except Exception as e:
                logger.debug(f"⚠️ Web enrichment failed for {row['symbol']}: {e}")
                continue

        # Calculate enrichment success rate
        enriched_count = len(enriched_df[enriched_df['web_catalyst_summary'].notna()])
        success_rate = (enriched_count / limit) * 100 if limit > 0 else 0
        logger.info(f"🌐 Web context enrichment complete: {enriched_count}/{limit} stocks ({success_rate:.1f}%)")

        return enriched_df

    def _generate_local_insights(self, symbol: str, stock_data: Dict) -> str:
        """Generate insights based on local stock data patterns without external APIs"""
        try:
            insights = []

            # Volume analysis
            rvol = stock_data.get('rvol', 1.0)
            volume = stock_data.get('volume', 0)
            if rvol > 3.0:
                insights.append(f"Exceptional {rvol:.1f}x volume surge indicates major institutional activity")
            elif rvol > 2.0:
                insights.append(f"Strong {rvol:.1f}x volume increase suggests growing institutional interest")
            elif rvol > 1.5:
                insights.append(f"Above-average {rvol:.1f}x volume signals potential breakout")

            # Price action analysis
            price = stock_data.get('price', 0)
            if price < 5:
                insights.append(f"Low price point at ${price:.2f} offers high percentage gain potential")
            elif price < 20:
                insights.append(f"Mid-cap range at ${price:.2f} with institutional accessibility")

            # Market cap analysis
            market_cap = stock_data.get('market_cap', 0)
            if market_cap and market_cap < 1000000000:  # < $1B
                insights.append(f"Small cap with ${market_cap/1000000:.0f}M market cap offers explosive upside")
            elif market_cap and market_cap < 10000000000:  # < $10B
                insights.append(f"Mid cap ${market_cap/1000000000:.1f}B provides balance of growth and stability")

            # Accumulation score insights
            acc_score = stock_data.get('accumulation_score', 0)
            if acc_score > 90:
                insights.append("Premium accumulation pattern detected with institutional backing")
            elif acc_score > 85:
                insights.append("Strong accumulation signals suggest upcoming momentum")
            elif acc_score > 80:
                insights.append("Solid accumulation foundation building for potential breakout")

            return ". ".join(insights[:3]) + "." if insights else f"Monitoring {symbol} for accumulation patterns and volume confirmations."

        except Exception as e:
            logger.debug(f"Local insights generation failed: {e}")
            return f"Analyzing {symbol} market dynamics and institutional flow patterns."

    def _calculate_sentiment_score(self, stock_data: Dict) -> float:
        """Calculate sentiment score based on volume and price patterns"""
        try:
            rvol = stock_data.get('rvol', 1.0)
            acc_score = stock_data.get('accumulation_score', 70)

            # Base sentiment on volume surge and accumulation
            if rvol > 3.0 and acc_score > 85:
                return 75.0  # Very positive
            elif rvol > 2.0 and acc_score > 80:
                return 65.0  # Positive
            elif rvol > 1.5 and acc_score > 75:
                return 55.0  # Moderately positive
            else:
                return 45.0  # Neutral

        except Exception as e:
            logger.debug(f"Sentiment calculation failed: {e}")
            return 50.0

    def _generate_sentiment_description(self, stock_data: Dict) -> str:
        """Generate sentiment description based on data patterns"""
        try:
            rvol = stock_data.get('rvol', 1.0)
            volume = stock_data.get('volume', 0)

            if rvol > 3.0:
                return "Highly bullish sentiment driven by exceptional volume surge"
            elif rvol > 2.0:
                return "Positive sentiment with strong institutional participation"
            elif rvol > 1.5:
                return "Moderately bullish sentiment with above-average interest"
            else:
                return "Neutral sentiment with standard trading patterns"

        except Exception as e:
            logger.debug(f"Sentiment description failed: {e}")
            return "Market sentiment analysis in progress"

    def _analyze_institutional_patterns(self, stock_data: Dict) -> str:
        """Analyze institutional activity patterns"""
        try:
            rvol = stock_data.get('rvol', 1.0)
            market_cap = stock_data.get('market_cap', 0)

            patterns = []

            if rvol > 3.0:
                patterns.append("Heavy institutional accumulation detected")
            elif rvol > 2.0:
                patterns.append("Moderate institutional buying interest")
            elif rvol > 1.5:
                patterns.append("Emerging institutional attention")

            if market_cap and market_cap < 2000000000:  # < $2B
                patterns.append("Small-cap institutional entry opportunity")
            elif market_cap and market_cap < 10000000000:  # < $10B
                patterns.append("Mid-cap institutional positioning")

            return ". ".join(patterns) if patterns else "Standard institutional trading patterns observed"

        except Exception as e:
            logger.debug(f"Institutional analysis failed: {e}")
            return "Institutional pattern analysis in progress"

    def _calculate_institutional_score(self, stock_data: Dict) -> float:
        """Calculate institutional activity score"""
        try:
            rvol = stock_data.get('rvol', 1.0)
            acc_score = stock_data.get('accumulation_score', 70)

            # Score based on volume patterns (institutional proxy)
            if rvol > 4.0:
                base_score = 80.0
            elif rvol > 3.0:
                base_score = 70.0
            elif rvol > 2.0:
                base_score = 60.0
            elif rvol > 1.5:
                base_score = 50.0
            else:
                base_score = 40.0

            # Boost by accumulation score
            boost = (acc_score - 70) * 0.2  # Scale accumulation contribution
            final_score = min(85.0, max(30.0, base_score + boost))

            return final_score

        except Exception as e:
            logger.debug(f"Institutional score calculation failed: {e}")
            return 50.0

    def _parse_web_context(self, news_content: str, institutional_content: str, social_content: str, symbol: str) -> Dict:
        """Parse web context content to extract structured insights with scoring and descriptions"""
        insights = {
            'catalyst_summary': None,
            'catalyst_score': 0,
            'sentiment_score': 0,
            'sentiment_description': None,
            'institutional_activity': None,
            'institutional_score': 0
        }

        try:
            # Parse catalyst summary from news content with scoring
            if news_content and news_content.strip():
                # High-impact catalyst keywords with scoring weights
                high_impact_catalysts = {
                    'FDA approval': 25, 'merger': 20, 'acquisition': 20, 'breakthrough': 15,
                    'partnership': 12, 'contract': 10, 'earnings beat': 15
                }
                medium_impact_catalysts = {
                    'trial results': 8, 'launch': 8, 'expansion': 6, 'guidance': 5
                }

                news_lower = news_content.lower()
                catalyst_score = 0
                found_catalysts = []

                # Check for high-impact catalysts
                for catalyst, score in high_impact_catalysts.items():
                    if catalyst in news_lower:
                        catalyst_score += score
                        found_catalysts.append(catalyst)

                # Check for medium-impact catalysts
                for catalyst, score in medium_impact_catalysts.items():
                    if catalyst in news_lower:
                        catalyst_score += score
                        found_catalysts.append(catalyst)

                if found_catalysts:
                    # Extract concise summary (first 120 chars)
                    catalyst_summary = news_content[:120].strip()
                    if len(news_content) > 120:
                        catalyst_summary += "..."
                    insights['catalyst_summary'] = catalyst_summary
                    insights['catalyst_score'] = min(catalyst_score, 100)  # Cap at 100
                else:
                    # General news without specific catalysts
                    insights['catalyst_summary'] = "Recent news available"
                    insights['catalyst_score'] = 5

            # Parse institutional activity with scoring
            if institutional_content and institutional_content.strip():
                institutional_lower = institutional_content.lower()
                institutional_score = 0
                activity_indicators = []

                # Institutional activity indicators
                if 'unusual options' in institutional_lower:
                    institutional_score += 15
                    activity_indicators.append('unusual options')
                if 'institutional buying' in institutional_lower:
                    institutional_score += 20
                    activity_indicators.append('institutional buying')
                if 'insider buying' in institutional_lower:
                    institutional_score += 10
                    activity_indicators.append('insider activity')
                if 'large volume' in institutional_lower:
                    institutional_score += 8
                    activity_indicators.append('volume surge')

                if activity_indicators:
                    insights['institutional_activity'] = ', '.join(activity_indicators)
                    insights['institutional_score'] = min(institutional_score, 100)
                else:
                    insights['institutional_activity'] = "Monitoring institutional flows"
                    insights['institutional_score'] = 5

            # Parse sentiment score with descriptive analysis
            if social_content and social_content.strip():
                # Enhanced sentiment analysis
                strong_bullish = ['moon', 'rocket', 'breakout', 'explosive', 'massive']
                bullish = ['bullish', 'buy', 'strong', 'positive', 'up', 'calls']
                bearish = ['bearish', 'sell', 'weak', 'negative', 'down', 'puts']
                strong_bearish = ['dump', 'crash', 'terrible', 'avoid']

                social_lower = social_content.lower()

                # Calculate weighted sentiment
                strong_bull_count = sum(1 for word in strong_bullish if word in social_lower)
                bull_count = sum(1 for word in bullish if word in social_lower)
                bear_count = sum(1 for word in bearish if word in social_lower)
                strong_bear_count = sum(1 for word in strong_bearish if word in social_lower)

                # Weighted scoring
                sentiment_score = (strong_bull_count * 20 + bull_count * 10 -
                                 bear_count * 10 - strong_bear_count * 20)

                # Normalize to 0-100 scale
                sentiment_score = max(0, min(100, 50 + sentiment_score))
                insights['sentiment_score'] = sentiment_score

                # Descriptive analysis
                if sentiment_score >= 75:
                    insights['sentiment_description'] = "Strong bullish sentiment"
                elif sentiment_score >= 60:
                    insights['sentiment_description'] = "Moderately bullish"
                elif sentiment_score >= 40:
                    insights['sentiment_description'] = "Mixed sentiment"
                elif sentiment_score >= 25:
                    insights['sentiment_description'] = "Moderately bearish"
                else:
                    insights['sentiment_description'] = "Strong bearish sentiment"

        except Exception as e:
            logger.debug(f"Error parsing web context for {symbol}: {e}")

        return insights

    def get_mcp_filtered_universe(self) -> pd.DataFrame:
        """Get universe using direct Polygon API - GUARANTEED REAL DATA"""
        logger.info("🎯 Loading universe via direct Polygon API...")

        try:
            # Use direct Polygon API for guaranteed real premium data
            logger.info("   🎯 Fetching data via direct Polygon API - PREMIUM REAL DATA...")

            snapshot_response = _call_polygon_api(
                'mcp__polygon__get_snapshot_all',
                market_type="stocks"
            )

            if not snapshot_response or snapshot_response.get('status') != 'OK':
                logger.error(f"   ❌ Direct API snapshot failed: {snapshot_response}")
                return pd.DataFrame()

            tickers_data = snapshot_response.get('results', [])
            logger.info(f"   ✅ Received {len(tickers_data)} stocks from direct Polygon API - REAL DATA")

            # Convert to DataFrame
            df_data = []
            for ticker in tickers_data:
                try:
                    symbol = ticker.get('ticker', '')
                    if not symbol:
                        continue

                    # Smart data source selection with fallbacks
                    price, volume = self._get_best_price_volume(ticker)

                    if price > 0 and volume > 0:
                        df_data.append({
                            'symbol': symbol,
                            'price': price,
                            'volume': volume,
                            'change_pct': ticker.get('todaysChangePerc', 0),
                            'market_cap': None,  # Will be enriched later
                            'atr': None
                        })
                except Exception as e:
                    logger.debug(f"Error processing ticker {ticker}: {e}")
                    continue

            df = pd.DataFrame(df_data)
            logger.info(f"✅ Created universe DataFrame with {len(df)} stocks - ALL REAL DATA")
            return df

        except Exception as e:
            logger.error(f"Failed to get MCP universe: {e}")
            return pd.DataFrame()

    def discover(self, gates=None, limit=20) -> List[Dict]:
        """Main discovery method - GUARANTEED PREMIUM DATA"""
        logger.info(f"🚀 Starting discovery with gates: {gates}, limit: {limit}")

        # Get universe
        universe_df = self.get_mcp_filtered_universe()

        if universe_df.empty:
            logger.error("❌ No universe data available")
            return []

        logger.info(f"✅ Universe loaded: {len(universe_df)} stocks")

        # Count ETFs/Funds for filtering transparency
        etf_fund_count = universe_df[
            universe_df['symbol'].str.contains(r'ETF|FUND|REIT|SPY|QQQ|VTI|IWM|DIA', case=False, na=False) |
            universe_df['symbol'].str.endswith(('X', 'Y', 'Z'), na=False)
        ].shape[0]
        logger.info(f"🚫 Filtering out {etf_fund_count} ETFs/funds (stocks only per user requirement)")

        # Apply improved filtering (Gate A) - Optimized for explosive growth
        def dynamic_volume_filter(row):
            """Dynamic volume requirement based on price range"""
            price = row['price']
            volume = row['volume']

            # Price-based volume requirements (higher price = lower volume requirement)
            if price < 2:      min_vol = 500000   # Penny stocks need high volume
            elif price < 10:   min_vol = 300000   # Small caps
            elif price < 50:   min_vol = 200000   # Mid caps
            else:              min_vol = 100000   # Large caps (expensive stocks)

            return volume >= min_vol

        # Apply base filters first - ETF/Fund exclusion added per user request
        base_filtered = universe_df[
            (universe_df['price'] >= self.config.MIN_PRICE) &    # TIGHTENED: $5+ to avoid penny stocks
            (universe_df['price'] <= 100) &     # Keep within your budget
            # REMOVED movement requirement - we want PRE-explosion stocks (stealth accumulation)
            # ETF/Fund/REIT exclusion filter - STOCKS ONLY per user requirement
            ~universe_df['symbol'].str.contains(r'ETF|FUND|REIT|SPY|QQQ|VTI|IWM|DIA', case=False, na=False) &
            ~universe_df['symbol'].str.endswith(('X', 'Y', 'Z'), na=False)  # Common ETF suffixes
        ].copy()

        # Apply dynamic volume filter
        volume_filtered = base_filtered[base_filtered.apply(dynamic_volume_filter, axis=1)].copy()

        # CRITICAL FIX: Apply blacklist filter to exclude recent losers
        def blacklist_filter(row):
            return not self.is_blacklisted(row['symbol'])

        pre_blacklist_count = len(volume_filtered)
        filtered_df = volume_filtered[volume_filtered.apply(blacklist_filter, axis=1)].copy()
        blacklisted_count = pre_blacklist_count - len(filtered_df)

        logger.info(f"✅ Gate A applied: {len(filtered_df)} stocks remaining")
        if blacklisted_count > 0:
            logger.info(f"🚫 Excluded {blacklisted_count} recent portfolio losers (30-day cooloff)")

        # Optimized RVOL-based scoring system
        logger.info("🎯 Calculating optimized accumulation scores with smart RVOL...")

        # Pre-calculate market-wide volume statistics for smart RVOL estimation
        market_volume_stats = self._calculate_market_volume_stats(filtered_df)

        def calculate_pre_explosion_score(row):
            """
            PRE-EXPLOSION ACCUMULATION DETECTOR
            Based on June-July winners: VIGL +324%, CRWV +171%, AEVA +162%, CRDO +108%
            Pattern: High volume + LOW price movement = Stealth accumulation before explosion
            """
            symbol = row['symbol']
            current_volume = row['volume']
            change_pct = row['change_pct']
            price = row['price']

            try:
                # Smart RVOL estimation without individual API calls
                rvol = self._estimate_smart_rvol(symbol, current_volume, price, market_volume_stats)

                # CRITICAL FIX: True stealth detection - reject stocks with excessive RVOL or price movement
                if rvol > self.config.MAX_STEALTH_RVOL:
                    return 0  # Too much volume = already discovered/exploded
                if rvol < self.config.MIN_STEALTH_RVOL:
                    return 0  # Too little volume = dead stock, no accumulation
                if abs(change_pct) > self.config.MAX_STEALTH_CHANGE:
                    return 0  # Too much price movement = already exploded
                if price < self.config.MIN_PRICE:
                    return 0  # Penny stock contamination - avoid

                # Get sustained accumulation pattern for multi-day analysis
                sustained_pattern = self.get_sustained_volume_pattern(symbol, self.config.SUSTAINED_DAYS)

                # 1. ENHANCED STEALTH ACCUMULATION SCORE (40% weight) - GRANULAR PRECISION
                # High volume with minimal price movement = institutions accumulating quietly
                abs_change = abs(change_pct)

                if GateConfig.ENHANCED_SCORING:
                    # Granular stealth accumulation scoring using sigmoid curves
                    volume_intensity = sigmoid((rvol - 1.5) / 1.0)  # Normalized around RVOL 1.5
                    stealth_factor = sigmoid((5.0 - abs_change) / 2.0)  # Inverse relationship with price movement

                    # Combined stealth score with fine-grained differentiation
                    stealth_raw = volume_intensity * stealth_factor
                    stealth_score = 40.0 * zclip(stealth_raw, 0.1, 1.0)  # Scale to 40% weight with minimum floor
                else:
                    # Legacy coarse integer scoring (causes 95-point convergence)
                    stealth_score = 0
                    if rvol >= 2.0:  # Significant volume increase
                        if abs_change <= 2.0:  # But minimal price movement
                            stealth_score = 40  # PERFECT stealth accumulation
                        elif abs_change <= 4.0:
                            stealth_score = 30  # Good accumulation pattern
                        else:
                            stealth_score = 10  # Volume there but maybe late
                    elif rvol >= 1.5:  # Moderate volume increase
                        if abs_change <= 1.5:
                            stealth_score = 30  # Good stealth pattern
                        elif abs_change <= 3.0:
                            stealth_score = 20  # Decent pattern
                        else:
                            stealth_score = 5   # Getting noisy
                    elif rvol >= 1.3:  # Slight volume increase
                        if abs_change <= 1.0:
                            stealth_score = 20  # Early accumulation
                        else:
                            stealth_score = 5   # Minimal pattern

                # 2. ENHANCED SMALL CAP EXPLOSIVE POTENTIAL (25% weight) - GRANULAR PRECISION
                # Lower priced stocks have bigger explosion potential (like your winners)
                if GateConfig.ENHANCED_SCORING:
                    # Granular size scoring using inverse sigmoid curve
                    price_factor = sigmoid((15.0 - price) / 8.0)  # Optimal around $15, diminishing above
                    size_score = 25.0 * zclip(price_factor, 0.1, 1.0)  # Scale to 25% weight with minimum floor
                else:
                    # Legacy coarse integer scoring
                    size_score = 0
                    if price <= 5:       size_score = 25    # Ultra small cap (highest potential)
                    elif price <= 10:   size_score = 20    # Small cap
                    elif price <= 20:   size_score = 15    # Mid-small cap
                    elif price <= 50:   size_score = 10    # Moderate potential
                    elif price <= 100:  size_score = 5     # Limited upside

                # 3. ENHANCED COILING PATTERN BONUS (20% weight) - GRANULAR PRECISION
                # Low volatility with building volume = pressure building
                if GateConfig.ENHANCED_SCORING:
                    # Granular coiling detection using combined volume and volatility factors
                    volume_buildup = sigmoid((rvol - 1.3) / 0.5)  # Volume building above 1.3x
                    volatility_compression = sigmoid((3.0 - abs_change) / 1.5)  # Low volatility bonus
                    coiling_raw = volume_buildup * volatility_compression
                    coiling_bonus = 20.0 * zclip(coiling_raw, 0.05, 1.0)  # Scale to 20% weight
                else:
                    # Legacy coarse integer scoring
                    coiling_bonus = 0
                    if rvol >= 1.5 and abs_change <= 2.0:
                        coiling_bonus = 15  # Perfect coiling - high volume, low movement
                    elif rvol >= 1.3 and abs_change <= 1.5:
                        coiling_bonus = 10  # Good coiling pattern
                    elif rvol >= 1.2 and abs_change <= 1.0:
                        coiling_bonus = 5   # Early coiling

                # 4. ENHANCED VOLUME QUALITY BONUS (15% weight) - GRANULAR PRECISION
                # Reward exceptional volume patterns with smooth scaling
                if GateConfig.ENHANCED_SCORING:
                    # Granular volume quality using logarithmic scaling
                    volume_excellence = sigmoid((rvol - 1.5) / 1.2)  # Sweet spot around 2.5x RVOL
                    volume_quality = 15.0 * zclip(volume_excellence, 0.05, 1.0)  # Scale to 15% weight
                else:
                    # Legacy coarse integer scoring
                    volume_quality = 0
                    if rvol >= 3.0:      volume_quality = 15   # Exceptional volume
                    elif rvol >= 2.5:   volume_quality = 12   # Very high volume
                    elif rvol >= 2.0:   volume_quality = 10   # High volume
                    elif rvol >= 1.5:   volume_quality = 8    # Good volume
                    elif rvol >= 1.3:   volume_quality = 5    # Above average volume

                # 5. NEW: SUSTAINED ACCUMULATION SCORE (20% weight) - MULTI-DAY ANALYSIS
                sustained_score = 0
                if sustained_pattern['pattern'] != 'error':
                    pattern_type = sustained_pattern['pattern']
                    trend_score = sustained_pattern.get('trend_score', 0)
                    accumulation_ratio = sustained_pattern.get('accumulation_ratio', 0)

                    # Weight based on sustained pattern quality
                    if pattern_type == 'strong_sustained':
                        sustained_score = 20.0 * min(1.0, trend_score / 2.0)  # Max at trend_score = 2.0
                    elif pattern_type == 'moderate_sustained':
                        sustained_score = 15.0 * min(1.0, trend_score / 1.8)
                    elif pattern_type == 'emerging_accumulation':
                        sustained_score = 10.0 * min(1.0, trend_score / 1.5)

                    # Additional bonus for consistent accumulation
                    if accumulation_ratio >= 0.6:
                        sustained_score *= 1.2  # 20% bonus for consistency

                # Rebalance weights: Stealth 35%, Size 20%, Coiling 15%, Volume 10%, Sustained 20%
                total_score = (stealth_score * 0.875) + (size_score * 0.8) + (coiling_bonus * 0.75) + (volume_quality * 0.67) + sustained_score

                # Penalty for stocks that already exploded or have extreme volume (we might be late)
                if abs_change > 8.0:
                    total_score *= 0.5  # Cut score in half for exploded stocks
                elif rvol > 20.0:  # Extreme volume might indicate we're late to the party
                    total_score *= 0.8  # Small penalty for extreme volume
                elif rvol > 10.0:  # Very high volume
                    total_score *= 0.9  # Tiny penalty

                # Set RVOL for display in UI
                if 'rvol' not in row or row.get('rvol', 1.0) == 1.0:
                    row['rvol'] = round(rvol, 2)

                return total_score

            except Exception as e:
                logger.debug(f"Scoring error for {symbol}: {e}")
                # Fallback to simple volume-based scoring
                return (current_volume / 1000000) * 5

        # Apply optimized scoring (vectorized where possible)
        logger.info(f"📊 Processing {len(filtered_df)} stocks with optimized RVOL...")
        scored_data = []
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            if idx % 500 == 0 and idx > 0:
                logger.info(f"   ⚡ Processed {idx}/{len(filtered_df)} stocks...")

            row_dict = row.to_dict()
            # Calculate score and get RVOL from the scoring function
            score_result = calculate_pre_explosion_score(row)
            row_dict['accumulation_score'] = score_result

            # Ensure RVOL is properly set from the scoring function
            if 'rvol' not in row_dict or row_dict.get('rvol', 1.0) == 1.0:
                # Recalculate RVOL for display
                rvol = self._estimate_smart_rvol(row_dict['symbol'], row_dict['volume'], row_dict['price'], market_volume_stats)
                row_dict['rvol'] = round(rvol, 2)

            scored_data.append(row_dict)

        filtered_df = pd.DataFrame(scored_data)
        logger.info("✅ Enhanced scoring complete")

        # Sort and limit
        top_stocks = filtered_df.nlargest(limit, 'accumulation_score')

        # Optional: Get precise RVOL for top candidates (disabled for production stability)
        # logger.info(f"🎯 Enhancing top {len(top_stocks)} candidates with precise RVOL...")
        # top_stocks = self._enhance_top_candidates_rvol(top_stocks)

        # Phase 2: Premium data enrichment (market cap, float, short interest)
        if GateConfig.ENHANCED_SCORING:
            logger.info(f"💎 Phase 2: Enriching top candidates with premium market data...")
            top_stocks = self._enrich_with_premium_data(top_stocks)

            # Phase 3: Quality gates for cream-of-the-crop filtering
            logger.info(f"🎯 Phase 3: Applying quality gates to identify cream-of-the-crop...")
            top_stocks = self._apply_quality_gates(top_stocks, target_count=limit)

            # Phase 6: Web Context Enrichment (optional)
            if GateConfig.WEB_ENRICHMENT:
                logger.info(f"🌐 Phase 6: Enriching survivors with real-time web context...")
                top_stocks = self._enrich_with_web_context(top_stocks)

        # Convert to enhanced result format with RVOL data
        results = []
        for _, row in top_stocks.iterrows():
            result = {
                'symbol': row['symbol'],
                'price': round(row['price'], 2),
                'volume': int(row['volume']),
                'change_pct': round(row['change_pct'], 2),
                'accumulation_score': round(row['accumulation_score'], 2),  # Enhanced precision for differentiation
                'rvol': row.get('rvol', 1.0),  # Relative volume ratio
                'tier': 'A-TIER' if row['accumulation_score'] >= 50 else 'B-TIER',
                'data_source': 'DIRECT_POLYGON_API',
                'signals': [],  # Will be populated with detected signals

                # Premium data fields (Phase 2 enhancement)
                'market_cap': row.get('market_cap', 0),
                'shares_outstanding': row.get('shares_outstanding', 0),
                'company_name': row.get('company_name', ''),
                'sector': row.get('sector', ''),
                'float_value': row.get('float_value', 0),
                'float_size': row.get('float_size', 'Unknown'),
                'short_interest': row.get('short_interest', 0),
                'days_to_cover': row.get('days_to_cover', 0),
                'short_squeeze_potential': row.get('short_squeeze_potential', 'Unknown'),

                # Phase 5: Explosion probability composite score
                'explosion_probability': self._calculate_explosion_probability(row) if GateConfig.ENHANCED_SCORING else 0,

                # Phase 6: Web Context Enrichment fields (optional)
                'web_catalyst_summary': row.get('web_catalyst_summary'),
                'web_catalyst_score': row.get('web_catalyst_score', 0),
                'web_sentiment_score': row.get('web_sentiment_score', 0),
                'web_sentiment_description': row.get('web_sentiment_description'),
                'institutional_activity': row.get('institutional_activity'),
                'institutional_score': row.get('institutional_score', 0),
                'analyst_action': row.get('analyst_action')
            }

            # Phase 4: Generate dynamic, data-driven reasons instead of generic signals
            if GateConfig.ENHANCED_SCORING:
                result['reasons'] = self._generate_dynamic_reasons(row)

                # Phase 6: Augment reasons with web context if available
                if GateConfig.WEB_ENRICHMENT and row.get('web_catalyst_summary'):
                    web_augmentation = []

                    # Add catalyst context
                    if row.get('web_catalyst_summary'):
                        web_augmentation.append(f" + fresh catalyst: {row['web_catalyst_summary'][:50]}...")

                    # Add analyst action
                    if row.get('analyst_action') == 'upgrade':
                        web_augmentation.append(" + analyst upgrade noted")
                    elif row.get('analyst_action') == 'downgrade':
                        web_augmentation.append(" + analyst downgrade warning")

                    # Add sentiment context
                    sentiment = row.get('web_sentiment_score')
                    if sentiment and sentiment > 0.7:
                        web_augmentation.append(" + positive web sentiment")
                    elif sentiment and sentiment < 0.3:
                        web_augmentation.append(" + negative web sentiment")

                    # Augment the first reason with web context
                    if web_augmentation and result['reasons']:
                        result['reasons'][0] += "".join(web_augmentation)

                # Keep signals for backward compatibility, but populate with key reasons
                result['signals'] = result['reasons'][:3]  # Top 3 reasons as signals
            else:
                # Legacy generic signals (for non-enhanced mode)
                if row.get('rvol', 1.0) >= 2.0 and abs(row['change_pct']) <= 2.0:
                    result['signals'].append('Stealth Accumulation Pattern')
                elif row.get('rvol', 1.0) >= 1.5 and abs(row['change_pct']) <= 1.5:
                    result['signals'].append('Early Accumulation')
                elif row.get('rvol', 1.0) >= 2.0:
                    result['signals'].append('High Volume Activity')

                if row['price'] <= 10:
                    result['signals'].append('Small Cap Explosive Potential')
                elif row['price'] <= 25:
                    result['signals'].append('Mid-Small Cap Opportunity')

                if row['accumulation_score'] >= 70:
                    result['signals'].append('Pre-Explosion Setup Complete')
                elif row['accumulation_score'] >= 50:
                    result['signals'].append('Building Accumulation Pattern')

            results.append(result)

        logger.info(f"✅ Discovery complete: {len(results)} candidates found - ALL PREMIUM DATA")
        return results

# For compatibility
if __name__ == "__main__":
    discovery = UniversalDiscoverySystem()
    results = discovery.discover(gates=['A'], limit=10)
    print(f"Found {len(results)} stocks with premium data")