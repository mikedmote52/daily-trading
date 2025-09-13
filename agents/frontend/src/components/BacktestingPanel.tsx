import React, { useState } from 'react';
import { Play, BarChart, TrendingUp, Calendar, Settings } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart as RechartsBarChart, Bar } from 'recharts';
import { useTrading } from '../contexts/TradingContext';

export const BacktestingPanel: React.FC = () => {
  const { state, runBacktest } = useTrading();
  const { backtestResults, isLoading } = state;
  const [selectedStrategy, setSelectedStrategy] = useState('momentum');
  const [parameters, setParameters] = useState({
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    initialCapital: 100000,
    maxPositions: 10,
    stopLoss: 0.05,
    takeProfit: 0.15
  });

  const strategies = [
    { id: 'momentum', name: 'Momentum Strategy', description: 'Buy stocks with strong price momentum' },
    { id: 'mean_reversion', name: 'Mean Reversion', description: 'Buy oversold stocks, sell overbought' },
    { id: 'breakout', name: 'Breakout Strategy', description: 'Trade stocks breaking resistance levels' },
    { id: 'ai_signals', name: 'AI Signals', description: 'Use AI-generated trading signals' }
  ];

  const handleRunBacktest = async () => {
    await runBacktest(selectedStrategy, parameters);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Mock performance data for charts
  const performanceData = [
    { date: '2024-01', portfolio: 100000, benchmark: 100000 },
    { date: '2024-02', portfolio: 105000, benchmark: 102000 },
    { date: '2024-03', portfolio: 98000, benchmark: 101000 },
    { date: '2024-04', portfolio: 112000, benchmark: 105000 },
    { date: '2024-05', portfolio: 108000, benchmark: 104000 },
    { date: '2024-06', portfolio: 125000, benchmark: 108000 }
  ];

  const monthlyReturns = [
    { month: 'Jan', returns: 5.0 },
    { month: 'Feb', returns: -6.7 },
    { month: 'Mar', returns: 14.3 },
    { month: 'Apr', returns: -3.6 },
    { month: 'May', returns: 15.7 },
    { month: 'Jun', returns: 2.1 }
  ];

  return (
    <div className="space-y-6">
      {/* Strategy Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Strategy Selection</h3>
          
          <div className="space-y-3 mb-6">
            {strategies.map((strategy) => (
              <div
                key={strategy.id}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                  selectedStrategy === strategy.id
                    ? 'border-blue-500 bg-blue-900/20'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
                onClick={() => setSelectedStrategy(strategy.id)}
              >
                <h4 className="font-medium text-white">{strategy.name}</h4>
                <p className="text-sm text-gray-400 mt-1">{strategy.description}</p>
              </div>
            ))}
          </div>
          
          <button
            onClick={handleRunBacktest}
            disabled={isLoading}
            className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg text-white font-medium transition-colors"
          >
            <Play className="h-5 w-5" />
            <span>{isLoading ? 'Running Backtest...' : 'Run Backtest'}</span>
          </button>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Settings className="h-5 w-5 mr-2" />
            Parameters
          </h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Start Date</label>
              <input
                type="date"
                value={parameters.startDate}
                onChange={(e) => setParameters(prev => ({ ...prev, startDate: e.target.value }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">End Date</label>
              <input
                type="date"
                value={parameters.endDate}
                onChange={(e) => setParameters(prev => ({ ...prev, endDate: e.target.value }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Initial Capital</label>
              <input
                type="number"
                value={parameters.initialCapital}
                onChange={(e) => setParameters(prev => ({ ...prev, initialCapital: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Max Positions</label>
              <input
                type="number"
                value={parameters.maxPositions}
                onChange={(e) => setParameters(prev => ({ ...prev, maxPositions: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Stop Loss (%)</label>
              <input
                type="number"
                step="0.01"
                value={parameters.stopLoss * 100}
                onChange={(e) => setParameters(prev => ({ ...prev, stopLoss: parseFloat(e.target.value) / 100 }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Take Profit (%)</label>
              <input
                type="number"
                step="0.01"
                value={parameters.takeProfit * 100}
                onChange={(e) => setParameters(prev => ({ ...prev, takeProfit: parseFloat(e.target.value) / 100 }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Backtest Results */}
      {backtestResults.length > 0 && (
        <div className="space-y-6">
          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {backtestResults.slice(0, 1).map((result) => (
              <React.Fragment key={result.strategy}>
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <p className="text-sm text-gray-400">Total Return</p>
                  <p className={`text-xl font-bold ${result.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatPercent(result.totalReturn)}
                  </p>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <p className="text-sm text-gray-400">Sharpe Ratio</p>
                  <p className="text-xl font-bold text-white">{result.sharpeRatio.toFixed(2)}</p>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <p className="text-sm text-gray-400">Max Drawdown</p>
                  <p className="text-xl font-bold text-red-400">{formatPercent(result.maxDrawdown)}</p>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <p className="text-sm text-gray-400">Win Rate</p>
                  <p className="text-xl font-bold text-white">{formatPercent(result.winRate)}</p>
                </div>
              </React.Fragment>
            ))}
          </div>

          {/* Performance Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <TrendingUp className="h-5 w-5 mr-2" />
                Cumulative Performance
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" tickFormatter={(value) => formatCurrency(value)} />
                  <Tooltip
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '6px'
                    }}
                    formatter={(value: number, name: string) => [
                      formatCurrency(value), 
                      name === 'portfolio' ? 'Strategy' : 'Benchmark'
                    ]}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="portfolio" 
                    stroke="#3B82F6" 
                    strokeWidth={3}
                    name="Strategy"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="benchmark" 
                    stroke="#6B7280" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Benchmark"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BarChart className="h-5 w-5 mr-2" />
                Monthly Returns
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsBarChart data={monthlyReturns}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="month" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" tickFormatter={(value) => `${value}%`} />
                  <Tooltip
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '6px'
                    }}
                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Monthly Return']}
                  />
                  <Bar 
                    dataKey="returns" 
                    fill={(dataPoint: any) => dataPoint.returns >= 0 ? '#10B981' : '#EF4444'}
                    radius={[4, 4, 0, 0]}
                  />
                </RechartsBarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Results Table */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">All Strategy Results</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Strategy</th>
                    <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Total Return</th>
                    <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Sharpe Ratio</th>
                    <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Max Drawdown</th>
                    <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Win Rate</th>
                    <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Total Trades</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Period</th>
                  </tr>
                </thead>
                <tbody>
                  {backtestResults.map((result, index) => (
                    <tr key={index} className="border-b border-gray-700">
                      <td className="py-3 px-4 text-sm font-medium text-white capitalize">
                        {result.strategy.replace('_', ' ')}
                      </td>
                      <td className={`py-3 px-4 text-sm text-right ${result.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatPercent(result.totalReturn)}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-300 text-right">
                        {result.sharpeRatio.toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-sm text-red-400 text-right">
                        {formatPercent(result.maxDrawdown)}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-300 text-right">
                        {formatPercent(result.winRate)}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-300 text-right">
                        {result.totalTrades}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-400">
                        {result.period}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};