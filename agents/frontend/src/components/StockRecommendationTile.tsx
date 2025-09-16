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

  // Format discovery reason
  const getDiscoveryReason = () => {
    const reasons = [];

    if (stockAnalysis.volume_score > 2) {
      reasons.push(`🚀 Volume surge ${stockAnalysis.volume_score.toFixed(1)}x`);
    }

    if (stockAnalysis.momentum_score > 10) {
      reasons.push(`📈 Strong momentum +${stockAnalysis.momentum_score.toFixed(1)}%`);
    }

    if (stockAnalysis.short_interest && stockAnalysis.short_interest > 15) {
      reasons.push(`🎯 Squeeze potential ${stockAnalysis.short_interest.toFixed(1)}% SI`);
    }

    if (stockAnalysis.signals.includes('Breakout')) {
      reasons.push('📊 Technical breakout');
    }

    if (stockAnalysis.volatility > 0.4) {
      reasons.push('⚡ High volatility energy');
    }

    return reasons.length > 0 ? reasons.join(' • ') : 'Multi-factor AI analysis';
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

      {/* Key Metrics */}
      <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
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
        <div className="mt-4 pt-4 border-t border-gray-600 text-xs text-gray-300 space-y-2">
          <div>
            <span className="text-gray-400">Momentum Score:</span> {stockAnalysis.momentum_score.toFixed(2)}
          </div>
          <div>
            <span className="text-gray-400">Volume Score:</span> {stockAnalysis.volume_score.toFixed(2)}x
          </div>
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
            <span className="text-gray-400">Recommendation:</span>
            <span className="text-green-400 font-bold"> {stockAnalysis.recommendation}</span>
          </div>
        </div>
      )}
    </div>
  );
};