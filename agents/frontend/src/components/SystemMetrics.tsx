import React from 'react';
import { Activity, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';
import { useAgentStatus } from '../contexts/AgentStatusContext';

export const SystemMetrics: React.FC = () => {
  const { state } = useAgentStatus();
  const { systemMetrics, isConnected } = state;

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'text-green-400';
      case 'degraded':
        return 'text-yellow-400';
      case 'critical':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4" />;
      case 'degraded':
        return <AlertCircle className="h-4 w-4" />;
      case 'critical':
        return <AlertCircle className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
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

  return (
    <div className="flex items-center space-x-6 text-sm">
      {/* Connection Status */}
      <div className="flex items-center space-x-2">
        <div
          className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-400' : 'bg-red-400'
          }`}
        />
        <span className="text-gray-300">
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {systemMetrics && (
        <>
          {/* System Health */}
          <div className="flex items-center space-x-2">
            <div className={getHealthColor(systemMetrics.systemHealth)}>
              {getHealthIcon(systemMetrics.systemHealth)}
            </div>
            <span className="text-gray-300">
              {systemMetrics.activeAgents}/{systemMetrics.totalAgents} Agents
            </span>
          </div>

          {/* Portfolio Value */}
          {systemMetrics.portfolioValue > 0 && (
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              <span className="text-gray-300">
                {formatCurrency(systemMetrics.portfolioValue)}
              </span>
            </div>
          )}

          {/* Daily P&L */}
          {systemMetrics.dailyPnl !== 0 && (
            <div className="flex items-center space-x-2">
              <span
                className={
                  systemMetrics.dailyPnl >= 0 ? 'text-green-400' : 'text-red-400'
                }
              >
                {formatCurrency(systemMetrics.dailyPnl)}
              </span>
              <span className="text-gray-400">
                ({formatPercent((systemMetrics.dailyPnl / systemMetrics.portfolioValue) * 100)})
              </span>
            </div>
          )}

          {/* Total Trades */}
          {systemMetrics.totalTrades > 0 && (
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-purple-400" />
              <span className="text-gray-300">
                {systemMetrics.totalTrades} Trades
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
};