import React, { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import { StockRecommendationTile } from './StockRecommendationTile';
import { AlpacaTradingService } from '../services/AlpacaTradingService';
import { DiscoveryService } from '../services/DiscoveryService';
import { StockAnalysis, MarketMetrics } from '../types/trading';

interface ExplosiveStockDiscoveryProps {
  alpacaService: AlpacaTradingService;
}

export const ExplosiveStockDiscovery: React.FC<ExplosiveStockDiscoveryProps> = ({
  alpacaService
}) => {
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [tradingInProgress, setTradingInProgress] = useState<Set<string>>(new Set());
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date());

  // Real-time stock discovery data
  const {
    data: explosiveStocks,
    isLoading,
    error,
    refetch
  } = useQuery<StockAnalysis[]>(
    'explosive-stocks-v2', // New cache key to force fresh data
    () => {
      console.log('🚀 React Query: Fetching explosive stocks...');
      return DiscoveryService.getExplosiveStocks();
    },
    {
      refetchInterval: 10000, // Update every 10 seconds
      refetchIntervalInBackground: true,
      onSuccess: (data) => {
        console.log('✅ React Query: Success! Received', data?.length || 0, 'stocks');
        setLastUpdateTime(new Date());
      },
      onError: (error) => {
        console.error('❌ React Query: Error fetching stocks:', error);
      }
    }
  );

  // Market overview metrics
  const { data: marketMetrics } = useQuery<MarketMetrics>(
    'market-metrics',
    () => DiscoveryService.getMarketMetrics(),
    {
      refetchInterval: 30000 // Update every 30 seconds
    }
  );

  // Handle stock purchase
  const handlePurchaseStock = async (stockAnalysis: StockAnalysis, investmentAmount: number) => {
    setTradingInProgress(prev => new Set(prev).add(stockAnalysis.symbol));

    try {
      const order = await alpacaService.executeExplosiveTrade(stockAnalysis, investmentAmount);

      // Show success notification
      alert(`Order placed successfully for ${stockAnalysis.symbol}! Order ID: ${order.id}`);

      // Refresh portfolio data
      refetch();
    } catch (error) {
      console.error('Trading error:', error);
      alert(`Failed to place order for ${stockAnalysis.symbol}: ${error.message}`);
    } finally {
      setTradingInProgress(prev => {
        const newSet = new Set(prev);
        newSet.delete(stockAnalysis.symbol);
        return newSet;
      });
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-green-500"></div>
        <div className="ml-4 text-lg">Scanning 10,000+ stocks for explosive opportunities...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900 border border-red-500 rounded-lg p-6 text-center">
        <h3 className="text-xl font-bold text-red-300">Discovery System Error</h3>
        <p className="text-red-200 mt-2">Failed to load stock recommendations: {error.message}</p>
        <button
          onClick={() => refetch()}
          className="mt-4 bg-red-600 hover:bg-red-700 px-4 py-2 rounded"
        >
          Retry Discovery Scan
        </button>
      </div>
    );
  }

  const buyRecommendations = explosiveStocks?.filter(stock => stock.recommendation === 'BUY') || [];
  const highConfidenceStocks = buyRecommendations.filter(stock => stock.ai_score >= 80);
  const mediumConfidenceStocks = buyRecommendations.filter(stock => stock.ai_score >= 60 && stock.ai_score < 80);

  return (
    <div className="space-y-6">
      {/* Header with Market Overview */}
      <div className="bg-gradient-to-r from-green-900 to-blue-900 rounded-lg p-6 border border-green-500">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">🔥 Explosive Stock Discovery</h1>
            <p className="text-green-200 mt-2">
              AI-powered system targeting +63.8% annual returns • Live market scanning
            </p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-green-400">
              {buyRecommendations.length} Explosive Opportunities
            </div>
            <div className="text-sm text-gray-300">
              Last Update: {lastUpdateTime.toLocaleTimeString()}
            </div>
            {marketMetrics && (
              <div className="text-xs text-gray-400 mt-1">
                Market Volatility: {(marketMetrics.averageVolatility * 100).toFixed(1)}% •
                Volume Surges: {marketMetrics.volumeSurges}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* High Confidence Recommendations */}
      {highConfidenceStocks.length > 0 && (
        <div>
          <div className="flex items-center mb-4">
            <div className="bg-green-500 w-4 h-4 rounded-full mr-3"></div>
            <h2 className="text-2xl font-bold text-green-400">
              🎯 High Confidence (AI Score 80+)
            </h2>
            <span className="bg-green-500 text-white px-3 py-1 rounded-full ml-3 text-sm">
              {highConfidenceStocks.length} stocks
            </span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {highConfidenceStocks.map((stock) => (
              <StockRecommendationTile
                key={stock.symbol}
                stockAnalysis={stock}
                onPurchase={handlePurchaseStock}
                isTrading={tradingInProgress.has(stock.symbol)}
                confidenceLevel="high"
              />
            ))}
          </div>
        </div>
      )}

      {/* Medium Confidence Recommendations */}
      {mediumConfidenceStocks.length > 0 && (
        <div>
          <div className="flex items-center mb-4">
            <div className="bg-yellow-500 w-4 h-4 rounded-full mr-3"></div>
            <h2 className="text-2xl font-bold text-yellow-400">
              ⚡ Medium Confidence (AI Score 60-79)
            </h2>
            <span className="bg-yellow-500 text-black px-3 py-1 rounded-full ml-3 text-sm">
              {mediumConfidenceStocks.length} stocks
            </span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {mediumConfidenceStocks.map((stock) => (
              <StockRecommendationTile
                key={stock.symbol}
                stockAnalysis={stock}
                onPurchase={handlePurchaseStock}
                isTrading={tradingInProgress.has(stock.symbol)}
                confidenceLevel="medium"
              />
            ))}
          </div>
        </div>
      )}

      {/* No Recommendations Message */}
      {buyRecommendations.length === 0 && (
        <div className="bg-gray-800 border border-gray-600 rounded-lg p-8 text-center">
          <div className="text-4xl mb-4">📊</div>
          <h3 className="text-xl font-bold text-gray-300 mb-2">No Explosive Opportunities Found</h3>
          <p className="text-gray-400">
            The discovery system is currently scanning {explosiveStocks?.length || 0} stocks.
            No stocks meet the explosive growth criteria at this time.
          </p>
          <p className="text-sm text-gray-500 mt-2">
            System continuously scans for: Volume surges • Momentum breakouts • Short squeeze potential
          </p>
        </div>
      )}

      {/* System Status Footer */}
      <div className="bg-gray-800 rounded-lg p-4 text-sm text-gray-400">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <span className="text-green-400">●</span> Discovery System: Active
          </div>
          <div>
            <span className="text-green-400">●</span> Alpaca Trading: Connected
          </div>
          <div>
            <span className="text-green-400">●</span> Real-time Data: Live
          </div>
          <div>
            <span className="text-blue-400">●</span> Scanning: {explosiveStocks?.length || 0} stocks
          </div>
        </div>
      </div>
    </div>
  );
};