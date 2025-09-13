import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Star, Plus, Search } from 'lucide-react';
import { useTrading } from '../contexts/TradingContext';

export const WatchlistPanel: React.FC = () => {
  const { state } = useTrading();
  const { watchlist } = state;
  const [searchTerm, setSearchTerm] = useState('');

  const filteredWatchlist = watchlist.filter(stock =>
    stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    stock.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const getAIScoreColor = (score?: number) => {
    if (!score) return 'text-gray-400';
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Header and Search */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
        <h2 className="text-2xl font-bold text-white">Stock Watchlist</h2>
        
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="h-5 w-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search stocks..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-colors">
            <Plus className="h-4 w-4" />
            <span>Add Stock</span>
          </button>
        </div>
      </div>

      {/* Watchlist Grid */}
      {filteredWatchlist.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredWatchlist.map((stock) => (
            <div key={stock.symbol} className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-gray-600 transition-colors">
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-bold text-white">{stock.symbol}</h3>
                  <p className="text-sm text-gray-400 truncate">{stock.name}</p>
                </div>
                <button className="text-gray-400 hover:text-yellow-400 transition-colors">
                  <Star className="h-5 w-5" />
                </button>
              </div>

              {/* Price Information */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold text-white">
                    {formatCurrency(stock.price)}
                  </span>
                  <div className={`flex items-center space-x-1 ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {stock.change >= 0 ? (
                      <TrendingUp className="h-4 w-4" />
                    ) : (
                      <TrendingDown className="h-4 w-4" />
                    )}
                    <span className="font-medium">
                      {formatCurrency(Math.abs(stock.change))} ({Math.abs(stock.changePercent).toFixed(2)}%)
                    </span>
                  </div>
                </div>

                {/* Additional Metrics */}
                <div className="grid grid-cols-2 gap-4 pt-3 border-t border-gray-700">
                  <div>
                    <p className="text-xs text-gray-400 uppercase">Volume</p>
                    <p className="text-sm text-white font-medium">{formatVolume(stock.volume)}</p>
                  </div>
                  
                  {stock.pe && (
                    <div>
                      <p className="text-xs text-gray-400 uppercase">P/E Ratio</p>
                      <p className="text-sm text-white font-medium">{stock.pe.toFixed(1)}</p>
                    </div>
                  )}
                  
                  {stock.shortInterest && (
                    <div>
                      <p className="text-xs text-gray-400 uppercase">Short Interest</p>
                      <p className="text-sm text-white font-medium">{stock.shortInterest.toFixed(1)}%</p>
                    </div>
                  )}
                  
                  {stock.aiScore && (
                    <div>
                      <p className="text-xs text-gray-400 uppercase">AI Score</p>
                      <p className={`text-sm font-medium ${getAIScoreColor(stock.aiScore)}`}>
                        {stock.aiScore}/100
                      </p>
                    </div>
                  )}
                </div>

                {/* AI Signals */}
                {stock.signals && stock.signals.length > 0 && (
                  <div className="pt-3 border-t border-gray-700">
                    <p className="text-xs text-gray-400 uppercase mb-2">AI Signals</p>
                    <div className="flex flex-wrap gap-1">
                      {stock.signals.map((signal, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-blue-600 text-blue-100 text-xs rounded-full"
                        >
                          {signal}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-2 mt-4 pt-4 border-t border-gray-700">
                <button className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-white text-sm font-medium transition-colors">
                  Buy
                </button>
                <button className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 rounded text-white text-sm font-medium transition-colors">
                  Sell
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="bg-gray-800 rounded-lg p-8 border border-gray-700">
            {searchTerm ? (
              <div>
                <p className="text-gray-400 text-lg mb-2">No stocks found matching "{searchTerm}"</p>
                <p className="text-gray-500">Try adjusting your search criteria</p>
              </div>
            ) : (
              <div>
                <p className="text-gray-400 text-lg mb-2">Your watchlist is empty</p>
                <p className="text-gray-500 mb-4">Add stocks to start tracking their performance</p>
                <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-colors">
                  Add Your First Stock
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Watchlist Stats */}
      {filteredWatchlist.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-sm text-gray-400">Total Stocks</p>
            <p className="text-xl font-bold text-white">{filteredWatchlist.length}</p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-sm text-gray-400">Avg AI Score</p>
            <p className="text-xl font-bold text-white">
              {(filteredWatchlist
                .filter(s => s.aiScore)
                .reduce((sum, s) => sum + (s.aiScore || 0), 0) / 
                filteredWatchlist.filter(s => s.aiScore).length || 0
              ).toFixed(0)}
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-sm text-gray-400">Gainers</p>
            <p className="text-xl font-bold text-green-400">
              {filteredWatchlist.filter(s => s.change > 0).length}
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-sm text-gray-400">Losers</p>
            <p className="text-xl font-bold text-red-400">
              {filteredWatchlist.filter(s => s.change < 0).length}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};