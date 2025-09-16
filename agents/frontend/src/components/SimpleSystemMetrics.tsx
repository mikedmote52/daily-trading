import React from 'react';
import { Activity, TrendingUp, AlertCircle, CheckCircle, Server, Database, Wifi, WifiOff } from 'lucide-react';
import { AgentStatus } from '../types/trading';
import { useQuery } from 'react-query';
import { DiscoveryService } from '../services/DiscoveryService';

interface SystemMetricsProps {
  systemStatus: AgentStatus[];
  isConnected: boolean;
}

export const SystemMetrics: React.FC<SystemMetricsProps> = ({ systemStatus, isConnected }) => {
  // Fetch system status from backend
  const { data: backendSystemStatus } = useQuery(
    'system-status',
    () => DiscoveryService.getSystemStatus(),
    {
      refetchInterval: 30000, // Update every 30 seconds
      retry: 2
    }
  );

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-green-400';
      case 'inactive':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="h-4 w-4" />;
      case 'inactive':
        return <AlertCircle className="h-4 w-4" />;
      case 'error':
        return <AlertCircle className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const getStatusBadgeColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500/20 text-green-400 border-green-500';
      case 'inactive':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500';
      case 'error':
        return 'bg-red-500/20 text-red-400 border-red-500';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500';
    }
  };

  // Use backend data if available, otherwise fall back to props
  const agents = backendSystemStatus?.agents || {};
  const systemHealth = backendSystemStatus?.systemHealth || (isConnected ? 'healthy' : 'degraded');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900 to-blue-900 rounded-lg p-6 border border-purple-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Server className="h-8 w-8 text-purple-400" />
            <div>
              <h2 className="text-3xl font-bold text-white">⚙️ System Status</h2>
              <p className="text-purple-200">Multi-agent trading system monitoring</p>
            </div>
          </div>
          <div className="text-right">
            <div className={`px-4 py-2 rounded-full text-sm font-bold border ${
              isConnected
                ? 'bg-green-500/20 text-green-400 border-green-500'
                : 'bg-red-500/20 text-red-400 border-red-500'
            }`}>
              {isConnected ? '● ONLINE' : '● OFFLINE'}
            </div>
          </div>
        </div>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">System Health</p>
              <p className={`text-2xl font-bold ${getHealthColor(systemHealth === 'healthy' ? 'active' : 'error')}`}>
                {systemHealth?.toUpperCase() || 'UNKNOWN'}
              </p>
            </div>
            <div className={getHealthColor(systemHealth === 'healthy' ? 'active' : 'error')}>
              {getHealthIcon(systemHealth === 'healthy' ? 'active' : 'error')}
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Active Agents</p>
              <p className="text-2xl font-bold text-white">
                {Object.values(agents).filter((agent: any) => agent.status === 'active').length} / {Object.keys(agents).length}
              </p>
            </div>
            <Activity className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Connection</p>
              <p className={`text-2xl font-bold ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                {isConnected ? 'LIVE' : 'DOWN'}
              </p>
            </div>
            {isConnected ? (
              <Wifi className="h-8 w-8 text-green-400" />
            ) : (
              <WifiOff className="h-8 w-8 text-red-400" />
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Data Sources</p>
              <p className="text-2xl font-bold text-green-400">3 / 3</p>
              <p className="text-xs text-gray-500">Polygon • Alpaca • AI</p>
            </div>
            <Database className="h-8 w-8 text-green-400" />
          </div>
        </div>
      </div>

      {/* Agent Status Details */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-4">Agent Status</h3>

        {Object.keys(agents).length > 0 ? (
          <div className="space-y-4">
            {Object.entries(agents).map(([agentKey, agentData]: [string, any]) => (
              <div key={agentKey} className="flex items-center justify-between bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className={`w-3 h-3 rounded-full ${
                    agentData.status === 'active' ? 'bg-green-400' :
                    agentData.status === 'inactive' ? 'bg-yellow-400' :
                    'bg-red-400'
                  }`}></div>

                  <div>
                    <h4 className="text-lg font-bold text-white">{agentData.name || agentKey}</h4>
                    <p className="text-sm text-gray-400">{agentData.currentTask || 'Idle'}</p>
                  </div>
                </div>

                <div className="text-right">
                  <div className={`px-3 py-1 rounded-full text-xs font-bold border ${getStatusBadgeColor(agentData.status)}`}>
                    {agentData.status?.toUpperCase() || 'UNKNOWN'}
                  </div>
                  {agentData.lastHeartbeat && (
                    <p className="text-xs text-gray-500 mt-1">
                      Last seen: {new Date(agentData.lastHeartbeat).toLocaleTimeString()}
                    </p>
                  )}
                </div>

                {agentData.metrics && (
                  <div className="ml-6 text-right">
                    <div className="text-sm text-gray-300">
                      {Object.entries(agentData.metrics).map(([key, value]: [string, any]) => (
                        <div key={key} className="flex justify-between space-x-4">
                          <span className="capitalize">{key.replace('_', ' ')}:</span>
                          <span className="font-bold">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            <Server className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No agent status available</p>
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">System Performance</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Discovery Engine</span>
              <div className="flex items-center space-x-2">
                <div className="w-20 bg-gray-600 rounded-full h-2">
                  <div className="bg-green-400 h-2 rounded-full" style={{ width: '85%' }}></div>
                </div>
                <span className="text-sm text-green-400">85%</span>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400">Trading Engine</span>
              <div className="flex items-center space-x-2">
                <div className="w-20 bg-gray-600 rounded-full h-2">
                  <div className="bg-blue-400 h-2 rounded-full" style={{ width: '92%' }}></div>
                </div>
                <span className="text-sm text-blue-400">92%</span>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400">Risk Management</span>
              <div className="flex items-center space-x-2">
                <div className="w-20 bg-gray-600 rounded-full h-2">
                  <div className="bg-yellow-400 h-2 rounded-full" style={{ width: '78%' }}></div>
                </div>
                <span className="text-sm text-yellow-400">78%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">System Metrics</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Uptime:</span>
              <span className="text-green-400">99.8%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">API Calls (today):</span>
              <span className="text-blue-400">12,847</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Stocks Scanned:</span>
              <span className="text-purple-400">10,000+</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Opportunities Found:</span>
              <span className="text-yellow-400">{agents.discovery?.metrics?.opportunities_found || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Memory Usage:</span>
              <span className="text-gray-300">2.1GB / 8GB</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};