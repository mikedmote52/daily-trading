import React, { useState } from 'react';
import { StockAnalysis } from '../types/trading';

interface StockRecommendationTileProps {
  stockAnalysis: StockAnalysis;
  onPurchase: (stock: StockAnalysis, amount: number) => void;
  isTrading: boolean;
  confidenceLevel: 'high' | 'medium' | 'low';
}

export const StockRecommendationTile: React.FC<StockRecommendationTileProps> = ({
  stockAnalysis,
  onPurchase,
  isTrading,
  confidenceLevel
}) => {
  const [investmentAmount, setInvestmentAmount] = useState<number>(1000);
  const [showDetails, setShowDetails] = useState<boolean>(false);

  // Calculate potential shares and returns
  const potentialShares = Math.floor(investmentAmount / stockAnalysis.price);
  const targetPrice = stockAnalysis.price * (1 + (stockAnalysis.ai_score / 100 * 0.5)); // Conservative target based on AI score
  const potentialReturn = (targetPrice - stockAnalysis.price) * potentialShares;
  const returnPercentage = ((targetPrice - stockAnalysis.price) / stockAnalysis.price) * 100;

  // Get confidence styling
  const getConfidenceStyle = () => {
    switch (confidenceLevel) {
      case 'high':
        return {
          border: 'border-green-500',
          bg: 'bg-gradient-to-br from-green-900 to-green-800',
          glow: 'shadow-lg shadow-green-500/20',
          score: 'text-green-400'
        };
      case 'medium':
        return {
          border: 'border-yellow-500',
          bg: 'bg-gradient-to-br from-yellow-900 to-yellow-800',
          glow: 'shadow-lg shadow-yellow-500/20',
          score: 'text-yellow-400'
        };
      default:
        return {
          border: 'border-gray-500',
          bg: 'bg-gradient-to-br from-gray-900 to-gray-800',
          glow: 'shadow-lg shadow-gray-500/20',
          score: 'text-gray-400'
        };
    }
  };

  const style = getConfidenceStyle();

  // Format discovery reason with RVOL emphasis
  const getDiscoveryReason = () => {
    const reasons = [];

    // CRITICAL: Show RVOL first (most important metric)
    const rvol = stockAnalysis.rvol || 1.0;
    if (rvol >= 2.0) {
      reasons.push(`🚀 ${rvol.toFixed(1)}x volume surge - Strong accumulation`);
    } else if (rvol >= 1.5) {
      reasons.push(`📊 ${rvol.toFixed(1)}x volume increase - Stealth pattern`);
    }

    if (stockAnalysis.volume_score > 2) {
      reasons.push(`Volume score: ${stockAnalysis.volume_score.toFixed(1)}x`);
    }

    if (stockAnalysis.momentum_score > 10) {
      reasons.push(`Momentum: +${stockAnalysis.momentum_score.toFixed(1)}%`);
    }

    if (stockAnalysis.short_interest && stockAnalysis.short_interest > 15) {
      reasons.push(`🎯 Squeeze setup: ${stockAnalysis.short_interest.toFixed(1)}% SI`);
    }

    if (stockAnalysis.signals.includes('Breakout')) {
      reasons.push('📊 Breakout pattern detected');
    }

    return reasons.length > 0 ? reasons.join(' • ') : `${rvol.toFixed(1)}x RVOL - Accumulation pattern`;
  };

  const handlePurchase = () => {
    if (investmentAmount > 0) {
      onPurchase(stockAnalysis, investmentAmount);
    }
  };

  return (
    <div className={`${style.bg} ${style.border} ${style.glow} border-2 rounded-xl p-6 transition-all hover:scale-105`}>
      {/* Header with Stock Info */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-2xl font-bold text-white">{stockAnalysis.symbol}</h3>
          <p className="text-gray-300 text-sm truncate max-w-[200px]" title={stockAnalysis.name}>
            {stockAnalysis.name}
          </p>
        </div>
        <div className="text-right">
          <div className={`text-3xl font-bold ${style.score}`}>
            {stockAnalysis.ai_score}
          </div>
          <div className="text-xs text-gray-400">AI Score</div>
        </div>
      </div>

      {/* Price Information */}
      <div className="bg-black/20 rounded-lg p-4 mb-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-gray-400">Current Price</div>
            <div className="text-xl font-bold text-white">
              ${stockAnalysis.price.toFixed(2)}
            </div>
            <div className={`text-sm ${stockAnalysis.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stockAnalysis.change_percent >= 0 ? '+' : ''}{stockAnalysis.change_percent.toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Target Price</div>
            <div className="text-xl font-bold text-green-400">
              ${targetPrice.toFixed(2)}
            </div>
            <div className="text-sm text-green-400">
              +{returnPercentage.toFixed(1)}% potential
            </div>
          </div>
        </div>
      </div>

      {/* Discovery Reason */}
      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-1">Discovery Reason</div>
        <div className="text-sm text-white bg-black/20 rounded p-2">
          {getDiscoveryReason()}
        </div>
      </div>

      {/* Key Metrics - RVOL FIRST */}
      <div className="grid grid-cols-4 gap-2 mb-4 text-xs">
        <div className="bg-gradient-to-br from-green-900/40 to-green-800/40 rounded p-2 text-center border border-green-600/30">
          <div className="text-green-300 font-semibold">RVOL</div>
          <div className="text-white font-bold text-lg">
            {(stockAnalysis.rvol || 1.0).toFixed(1)}x
          </div>
        </div>
        <div className="bg-black/20 rounded p-2 text-center">
          <div className="text-gray-400">Volume</div>
          <div className="text-white font-bold">
            {(stockAnalysis.volume / 1000000).toFixed(1)}M
          </div>
        </div>
        <div className="bg-black/20 rounded p-2 text-center">
          <div className="text-gray-400">Volatility</div>
          <div className="text-white font-bold">
            {(stockAnalysis.volatility * 100).toFixed(0)}%
          </div>
        </div>
        <div className="bg-black/20 rounded p-2 text-center">
          <div className="text-gray-400">Market Cap</div>
          <div className="text-white font-bold">
            {stockAnalysis.market_cap ? `$${(stockAnalysis.market_cap / 1000000000).toFixed(1)}B` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Trading Patterns */}
      {stockAnalysis.signals.length > 0 && (
        <div className="mb-4">
          <div className="text-sm text-gray-400 mb-2">Detected Patterns</div>
          <div className="flex flex-wrap gap-1">
            {stockAnalysis.signals.map((signal, index) => (
              <span
                key={index}
                className="bg-blue-600 text-blue-100 text-xs px-2 py-1 rounded-full"
              >
                {signal}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Investment Calculator */}
      <div className="border-t border-gray-600 pt-4 mb-4">
        <div className="text-sm text-gray-400 mb-2">Investment Amount</div>
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-white">$</span>
          <input
            type="number"
            value={investmentAmount}
            onChange={(e) => setInvestmentAmount(Number(e.target.value))}
            className="bg-black/40 border border-gray-600 rounded px-3 py-1 text-white flex-1"
            min={100}
            step={100}
          />
        </div>
        <div className="text-xs text-gray-400">
          Shares: {potentialShares} • Potential Return:
          <span className="text-green-400 font-bold"> ${potentialReturn.toFixed(0)}</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="space-y-2">
        <button
          onClick={handlePurchase}
          disabled={isTrading || investmentAmount < 100}
          className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${
            isTrading
              ? 'bg-gray-600 cursor-not-allowed'
              : confidenceLevel === 'high'
              ? 'bg-green-600 hover:bg-green-700 hover:shadow-lg'
              : 'bg-yellow-600 hover:bg-yellow-700 hover:shadow-lg'
          }`}
        >
          {isTrading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Placing Order...
            </div>
          ) : (
            `🚀 Buy ${stockAnalysis.symbol} - $${investmentAmount}`
          )}
        </button>

        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full py-2 px-4 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-black/20 transition-all"
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {/* Detailed Analysis (Expandable) */}
      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-600 text-xs space-y-4">

          {/* Core Metrics */}
          <div className="bg-black/30 rounded-lg p-3">
            <h4 className="text-white font-bold mb-2">🎯 Core Analysis</h4>
            <div className="space-y-1 text-gray-300">
              <div>
                <span className="text-gray-400">AI Score:</span>
                <span className={`font-bold ml-1 ${stockAnalysis.ai_score >= 80 ? 'text-green-400' : stockAnalysis.ai_score >= 60 ? 'text-yellow-400' : 'text-gray-400'}`}>
                  {stockAnalysis.ai_score}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Momentum Score:</span> {stockAnalysis.momentum_score.toFixed(2)}
              </div>
              <div>
                <span className="text-gray-400">Volume Score:</span> {stockAnalysis.volume_score.toFixed(2)}x
              </div>
              {stockAnalysis.explosion_probability && (
                <div>
                  <span className="text-gray-400">Explosion Probability:</span>
                  <span className="text-orange-400 font-bold ml-1">{stockAnalysis.explosion_probability.toFixed(1)}%</span>
                </div>
              )}
            </div>
          </div>

          {/* Web Enrichment Insights */}
          {(stockAnalysis.web_catalyst_summary || stockAnalysis.web_sentiment_score || stockAnalysis.institutional_activity) && (
            <div className="bg-blue-900/30 rounded-lg p-3 border border-blue-700/30">
              <h4 className="text-blue-300 font-bold mb-2">🌐 Market Intelligence</h4>

              {/* Catalyst Analysis */}
              {stockAnalysis.web_catalyst_summary && (
                <div className="mb-3">
                  <div className="text-blue-200 font-medium text-xs mb-1">
                    📈 Catalyst Summary
                    {stockAnalysis.web_catalyst_score && (
                      <span className="ml-2 bg-blue-600 text-white px-2 py-0.5 rounded text-xs">
                        Score: {stockAnalysis.web_catalyst_score}
                      </span>
                    )}
                  </div>
                  <div className="text-gray-300 text-xs leading-relaxed bg-black/20 p-2 rounded">
                    {stockAnalysis.web_catalyst_summary}
                  </div>
                </div>
              )}

              {/* Sentiment Analysis */}
              {stockAnalysis.web_sentiment_score !== undefined && stockAnalysis.web_sentiment_score > 0 && (
                <div className="mb-3">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-blue-200 font-medium text-xs">💭 Market Sentiment</div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-1 ${
                        stockAnalysis.web_sentiment_score >= 70 ? 'bg-green-400' :
                        stockAnalysis.web_sentiment_score >= 50 ? 'bg-yellow-400' : 'bg-red-400'
                      }`}></div>
                      <span className="text-xs text-gray-300">{stockAnalysis.web_sentiment_score.toFixed(1)}/100</span>
                    </div>
                  </div>
                  {stockAnalysis.web_sentiment_description && (
                    <div className="text-gray-300 text-xs bg-black/20 p-2 rounded">
                      {stockAnalysis.web_sentiment_description}
                    </div>
                  )}
                </div>
              )}

              {/* Institutional Activity */}
              {stockAnalysis.institutional_activity && (
                <div className="mb-2">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-blue-200 font-medium text-xs">🏦 Institutional Activity</div>
                    {stockAnalysis.institutional_score && (
                      <span className="bg-purple-600 text-white px-2 py-0.5 rounded text-xs">
                        Score: {stockAnalysis.institutional_score.toFixed(1)}
                      </span>
                    )}
                  </div>
                  <div className="text-gray-300 text-xs bg-black/20 p-2 rounded">
                    {stockAnalysis.institutional_activity}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Technical Details */}
          <div className="bg-gray-800/50 rounded-lg p-3">
            <h4 className="text-gray-300 font-bold mb-2">📊 Technical Data</h4>
            <div className="grid grid-cols-2 gap-2 text-gray-300">
              {stockAnalysis.short_interest && (
                <div>
                  <span className="text-gray-400">Short Interest:</span> {stockAnalysis.short_interest.toFixed(1)}%
                </div>
              )}
              {stockAnalysis.pe_ratio && (
                <div>
                  <span className="text-gray-400">P/E Ratio:</span> {stockAnalysis.pe_ratio.toFixed(1)}
                </div>
              )}
              <div>
                <span className="text-gray-400">Volatility:</span> {(stockAnalysis.volatility * 100).toFixed(0)}%
              </div>
              <div>
                <span className="text-gray-400">Recommendation:</span>
                <span className="text-green-400 font-bold ml-1">{stockAnalysis.recommendation}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};