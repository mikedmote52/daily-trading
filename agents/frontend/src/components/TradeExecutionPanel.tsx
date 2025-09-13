import React, { useState } from 'react';
import { Send, Clock, CheckCircle, XCircle, TrendingUp } from 'lucide-react';
import { useTrading } from '../contexts/TradingContext';

export const TradeExecutionPanel: React.FC = () => {
  const { state, executeTradeCommand } = useTrading();
  const { trades, isLoading, error } = state;
  const [command, setCommand] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!command.trim()) return;

    try {
      await executeTradeCommand(command);
      setCommandHistory(prev => [command, ...prev.slice(0, 9)]); // Keep last 10 commands
      setCommand('');
    } catch (err) {
      console.error('Trade execution error:', err);
    }
  };

  const getTradeStatusIcon = (status: string) => {
    switch (status) {
      case 'filled':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'cancelled':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-400" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatDateTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const recentTrades = trades.slice(0, 10); // Show last 10 trades

  const sampleCommands = [
    'buy 100 AAPL at market',
    'sell 50 TSLA at limit 250',
    'buy $1000 worth of SPY',
    'set stop loss on MSFT at 380',
    'close all positions'
  ];

  return (
    <div className="space-y-6">
      {/* Trade Command Interface */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">AI Trade Execution</h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="Enter trade command (e.g., 'buy 100 AAPL at market')"
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-12"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !command.trim()}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-blue-400 hover:text-blue-300 disabled:text-gray-500 disabled:cursor-not-allowed"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
          
          {error && (
            <div className="p-3 bg-red-900/20 border border-red-700 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}
          
          <div className="text-sm text-gray-400">
            <p className="mb-2">Sample commands:</p>
            <div className="flex flex-wrap gap-2">
              {sampleCommands.map((sample, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => setCommand(sample)}
                  className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </form>

        {/* Command History */}
        {commandHistory.length > 0 && (
          <div className="mt-6 pt-6 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-400 mb-3">Command History</h4>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {commandHistory.map((cmd, index) => (
                <button
                  key={index}
                  onClick={() => setCommand(cmd)}
                  className="block w-full text-left px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-300 transition-colors"
                >
                  {cmd}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Recent Trades */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">Recent Trades</h3>
          <div className="flex items-center space-x-2 text-sm text-gray-400">
            <TrendingUp className="h-4 w-4" />
            <span>{trades.length} total trades</span>
          </div>
        </div>

        {recentTrades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Symbol</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Side</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Quantity</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Price</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Value</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Time</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Strategy</th>
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((trade) => (
                  <tr key={trade.id} className="border-b border-gray-700 hover:bg-gray-700/50">
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        {getTradeStatusIcon(trade.status)}
                        <span className={`text-sm capitalize ${
                          trade.status === 'filled' ? 'text-green-400' :
                          trade.status === 'cancelled' ? 'text-red-400' :
                          'text-yellow-400'
                        }`}>
                          {trade.status}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm font-medium text-white">{trade.symbol}</td>
                    <td className="py-3 px-4">
                      <span className={`text-sm font-medium ${
                        trade.side === 'buy' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {trade.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{trade.quantity}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">
                      {formatCurrency(trade.price)}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">
                      {formatCurrency(trade.price * trade.quantity)}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-400">
                      {formatDateTime(trade.timestamp)}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-400">
                      {trade.strategy || 'Manual'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="text-gray-400">
              <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">No trades executed yet</p>
              <p className="text-sm">Use the command interface above to execute your first trade</p>
            </div>
          </div>
        )}
      </div>

      {/* Trading Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-sm text-gray-400">Total Trades</p>
          <p className="text-xl font-bold text-white">{trades.length}</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-sm text-gray-400">Filled Orders</p>
          <p className="text-xl font-bold text-green-400">
            {trades.filter(t => t.status === 'filled').length}
          </p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-sm text-gray-400">Pending Orders</p>
          <p className="text-xl font-bold text-yellow-400">
            {trades.filter(t => t.status === 'pending').length}
          </p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-sm text-gray-400">Success Rate</p>
          <p className="text-xl font-bold text-blue-400">
            {trades.length > 0 
              ? ((trades.filter(t => t.status === 'filled').length / trades.length) * 100).toFixed(1)
              : 0
            }%
          </p>
        </div>
      </div>
    </div>
  );
};