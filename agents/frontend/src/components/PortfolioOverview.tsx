import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useTrading } from '../contexts/TradingContext';

export const PortfolioOverview: React.FC = () => {
  const { state } = useTrading();
  const { positions, portfolioValue, dailyPnl } = state;

  // Mock data for charts - in production this would come from the backend
  const performanceData = [
    { time: '9:30', value: portfolioValue * 0.98 },
    { time: '10:30', value: portfolioValue * 1.01 },
    { time: '11:30', value: portfolioValue * 0.99 },
    { time: '12:30', value: portfolioValue * 1.02 },
    { time: '13:30', value: portfolioValue * 1.00 },
    { time: '14:30', value: portfolioValue * 1.03 },
    { time: '15:30', value: portfolioValue },
  ];

  const sectorData = positions.reduce((acc, position) => {
    // This would normally come from stock metadata
    const sector = 'Technology'; // Simplified
    const existing = acc.find(item => item.name === sector);
    if (existing) {
      existing.value += position.marketValue;
    } else {
      acc.push({ name: sector, value: position.marketValue });
    }
    return acc;
  }, [] as { name: string; value: number }[]);

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const totalUnrealizedPnl = positions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);

  return (
    <div className="space-y-6">
      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Portfolio Value</p>
              <p className="text-2xl font-bold text-white">{formatCurrency(portfolioValue)}</p>
            </div>
            <DollarSign className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Daily P&L</p>
              <p className={`text-2xl font-bold ${dailyPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(dailyPnl)}
              </p>
            </div>
            {dailyPnl >= 0 ? (
              <TrendingUp className="h-8 w-8 text-green-400" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-400" />
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Unrealized P&L</p>
              <p className={`text-2xl font-bold ${totalUnrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(totalUnrealizedPnl)}
              </p>
            </div>
            <BarChart3 className="h-8 w-8 text-purple-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Positions</p>
              <p className="text-2xl font-bold text-white">{positions.length}</p>
            </div>
            <BarChart3 className="h-8 w-8 text-orange-400" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Chart */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Daily Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" tickFormatter={(value) => formatCurrency(value)} />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '6px'
                }}
                labelStyle={{ color: '#F9FAFB' }}
                formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Sector Allocation */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Sector Allocation</h3>
          {sectorData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={sectorData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {sectorData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-400">No positions to display</p>
            </div>
          )}
        </div>
      </div>

      {/* Current Positions */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Current Positions</h3>
        {positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-4 text-sm font-medium text-gray-400">Symbol</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Shares</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Avg Price</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Current Price</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Market Value</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Unrealized P&L</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">% Change</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <tr key={position.symbol} className="border-b border-gray-700">
                    <td className="py-3 px-4 text-sm font-medium text-white">{position.symbol}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{position.shares}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{formatCurrency(position.avgPrice)}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{formatCurrency(position.currentPrice)}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{formatCurrency(position.marketValue)}</td>
                    <td className={`py-3 px-4 text-sm text-right ${position.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatCurrency(position.unrealizedPnl)}
                    </td>
                    <td className={`py-3 px-4 text-sm text-right ${position.unrealizedPnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {position.unrealizedPnlPercent.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-400">No positions currently held</p>
          </div>
        )}
      </div>
    </div>
  );
};