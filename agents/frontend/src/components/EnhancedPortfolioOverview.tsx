import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { TrendingUp, TrendingDown, DollarSign, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlpacaTradingService } from '../services/AlpacaTradingService';
import { Position, AccountInfo, AlpacaOrder } from '../services/AlpacaTradingService';

interface PortfolioOverviewProps {
  alpacaService: AlpacaTradingService;
}

export const PortfolioOverview: React.FC<PortfolioOverviewProps> = ({ alpacaService }) => {
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [showOrders, setShowOrders] = useState<boolean>(false);

  // Fetch account information
  const { data: accountInfo, isLoading: accountLoading } = useQuery<AccountInfo>(
    'account-info',
    () => alpacaService.getAccountInfo(),
    {
      refetchInterval: 30000, // Update every 30 seconds
      retry: 3
    }
  );

  // Fetch current positions
  const { data: positions, isLoading: positionsLoading } = useQuery<Position[]>(
    'positions',
    () => alpacaService.getPositions(),
    {
      refetchInterval: 10000, // Update every 10 seconds
      retry: 3
    }
  );

  // Fetch recent orders
  const { data: recentOrders } = useQuery<AlpacaOrder[]>(
    'recent-orders',
    () => alpacaService.getOrders(20),
    {
      refetchInterval: 15000 // Update every 15 seconds
    }
  );

  // Handle position closure
  const handleClosePosition = async (symbol: string) => {
    if (!window.confirm(`Are you sure you want to close your ${symbol} position?`)) {
      return;
    }

    try {
      await alpacaService.closePosition(symbol);
      alert(`Position ${symbol} closed successfully`);
      // Refresh data
      window.location.reload();
    } catch (error: any) {
      alert(`Failed to close position ${symbol}: ${error.message}`);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  if (accountLoading || positionsLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse bg-gray-800 h-32 rounded-lg"></div>
        <div className="animate-pulse bg-gray-800 h-64 rounded-lg"></div>
      </div>
    );
  }

  const totalPositionValue = positions?.reduce((sum, pos) => sum + pos.market_value, 0) || 0;
  const totalUnrealizedPL = positions?.reduce((sum, pos) => sum + pos.unrealized_pl, 0) || 0;
  const totalUnrealizedPLPercent = totalPositionValue > 0 ? (totalUnrealizedPL / (totalPositionValue - totalUnrealizedPL)) * 100 : 0;

  // Create sector allocation data from positions
  const sectorData = positions?.reduce((acc, position) => {
    // Simple sector classification - in production, you'd get this from a data service
    const sector = classifySymbolSector(position.symbol);
    const existing = acc.find(item => item.name === sector);
    if (existing) {
      existing.value += position.market_value;
    } else {
      acc.push({ name: sector, value: position.market_value });
    }
    return acc;
  }, [] as { name: string; value: number }[]) || [];

  // Mock performance data - in production, this would come from historical data
  const performanceData = [
    { time: '9:30', value: accountInfo ? accountInfo.portfolio_value * 0.98 : 0 },
    { time: '10:30', value: accountInfo ? accountInfo.portfolio_value * 1.01 : 0 },
    { time: '11:30', value: accountInfo ? accountInfo.portfolio_value * 0.99 : 0 },
    { time: '12:30', value: accountInfo ? accountInfo.portfolio_value * 1.02 : 0 },
    { time: '13:30', value: accountInfo ? accountInfo.portfolio_value * 1.00 : 0 },
    { time: '14:30', value: accountInfo ? accountInfo.portfolio_value * 1.03 : 0 },
    { time: '15:30', value: accountInfo ? accountInfo.portfolio_value : 0 },
  ];

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  return (
    <div className="space-y-6">
      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Portfolio Value</p>
              <p className="text-2xl font-bold text-white">
                {formatCurrency(accountInfo?.portfolio_value || 0)}
              </p>
              <p className="text-xs text-blue-400">
                Cash: {formatCurrency(accountInfo?.cash || 0)}
              </p>
            </div>
            <DollarSign className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Buying Power</p>
              <p className="text-2xl font-bold text-green-400">
                {formatCurrency(accountInfo?.buying_power || 0)}
              </p>
              <p className="text-xs text-gray-400">Available to trade</p>
            </div>
            <DollarSign className="h-8 w-8 text-green-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Unrealized P&L</p>
              <p className={`text-2xl font-bold ${totalUnrealizedPL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(totalUnrealizedPL)}
              </p>
              <p className={`text-xs ${totalUnrealizedPLPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {totalUnrealizedPLPercent >= 0 ? '+' : ''}{totalUnrealizedPLPercent.toFixed(2)}%
              </p>
            </div>
            {totalUnrealizedPL >= 0 ? (
              <TrendingUp className="h-8 w-8 text-green-400" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-400" />
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Active Positions</p>
              <p className="text-2xl font-bold text-white">{positions?.length || 0}</p>
              <p className="text-xs text-gray-400">
                Day Trades: {accountInfo?.day_trade_count || 0}
              </p>
            </div>
            <BarChart3 className="h-8 w-8 text-orange-400" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Chart */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Portfolio Performance</h3>
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
          <h3 className="text-lg font-semibold text-white mb-4">Position Allocation</h3>
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

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <button
          onClick={() => setShowOrders(!showOrders)}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors"
        >
          {showOrders ? 'Hide' : 'Show'} Recent Orders
        </button>
      </div>

      {/* Current Positions */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Current Positions</h3>
        {positions && positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-4 text-sm font-medium text-gray-400">Symbol</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Shares</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Cost Basis</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Market Value</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Unrealized P&L</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">% Change</th>
                  <th className="text-center py-2 px-4 text-sm font-medium text-gray-400">Actions</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <tr key={position.symbol} className="border-b border-gray-700 hover:bg-gray-700">
                    <td className="py-3 px-4 text-sm font-medium text-white">{position.symbol}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{position.qty}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{formatCurrency(position.cost_basis)}</td>
                    <td className="py-3 px-4 text-sm text-gray-300 text-right">{formatCurrency(position.market_value)}</td>
                    <td className={`py-3 px-4 text-sm text-right ${position.unrealized_pl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatCurrency(position.unrealized_pl)}
                    </td>
                    <td className={`py-3 px-4 text-sm text-right ${position.unrealized_plpc >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {position.unrealized_plpc >= 0 ? '+' : ''}{(position.unrealized_plpc * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-center">
                      <button
                        onClick={() => handleClosePosition(position.symbol)}
                        className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-xs transition-colors"
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-400 text-lg">No positions currently held</p>
            <p className="text-gray-500 text-sm mt-2">Start trading explosive stocks to see positions here</p>
          </div>
        )}
      </div>

      {/* Recent Orders (Collapsible) */}
      {showOrders && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Orders</h3>
          {recentOrders && recentOrders.length > 0 ? (
            <div className="space-y-3">
              {recentOrders.slice(0, 10).map((order, index) => (
                <div key={order.id || index} className="flex justify-between items-center bg-gray-700 p-3 rounded hover:bg-gray-600">
                  <div className="flex items-center space-x-4">
                    <span className="font-bold text-white">{order.symbol}</span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      order.side === 'buy' ? 'bg-green-600' : 'bg-red-600'
                    } text-white`}>
                      {order.side.toUpperCase()}
                    </span>
                    <span className="text-gray-300">{order.qty} shares</span>
                    <span className="text-gray-400 text-sm">{order.order_type}</span>
                  </div>

                  <div className="text-right">
                    <p className={`font-bold text-sm ${
                      order.status === 'filled' ? 'text-green-400' :
                      order.status === 'cancelled' ? 'text-red-400' :
                      'text-yellow-400'
                    }`}>
                      {order.status.toUpperCase()}
                    </p>
                    {order.filled_price && (
                      <p className="text-gray-400 text-sm">
                        ${order.filled_price.toFixed(2)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <p>No recent orders</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Helper function to classify sectors
function classifySymbolSector(symbol: string): string {
  const tech = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 'CRM', 'ADBE', 'ORCL'];
  const healthcare = ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'LLY', 'AMGN'];
  const finance = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SPGI', 'AXP', 'TFC'];
  const energy = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'PXD', 'OKE'];

  const upper = symbol.toUpperCase();

  if (tech.includes(upper) || upper.includes('TECH') || upper.includes('AI') || upper.includes('SOFT')) {
    return 'Technology';
  } else if (healthcare.includes(upper) || upper.includes('BIO') || upper.includes('PHARM')) {
    return 'Healthcare';
  } else if (finance.includes(upper) || upper.includes('BANK') || upper.includes('FIN')) {
    return 'Financials';
  } else if (energy.includes(upper) || upper.includes('OIL') || upper.includes('GAS')) {
    return 'Energy';
  } else {
    return 'Other';
  }
}