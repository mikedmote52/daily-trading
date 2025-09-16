import React, { useState, useEffect } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ExplosiveStockDiscovery } from './ExplosiveStockDiscovery';
import { PortfolioOverview } from './EnhancedPortfolioOverview';
import { SystemMetrics } from './SimpleSystemMetrics';
import { AlertsPanel } from './SimpleAlertsPanel';
import { AlpacaTradingService } from '../services/AlpacaTradingService';
import { DiscoveryService } from '../services/DiscoveryService';
import { TradingAlert, AgentStatus } from '../types/trading';

// Configuration from environment variables
const ALPACA_CONFIG = {
  apiKey: process.env.REACT_APP_ALPACA_API_KEY || '',
  secretKey: process.env.REACT_APP_ALPACA_SECRET_KEY || '',
  isLiveTrading: process.env.REACT_APP_ALPACA_LIVE_TRADING === 'true'
};

// Create React Query client for caching and real-time updates
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 10000, // Refetch every 10 seconds
      refetchIntervalInBackground: true,
      retry: 3,
      staleTime: 5000 // Consider data stale after 5 seconds
    }
  }
});

export const TradingDashboard: React.FC = () => {
  const [alpacaService] = useState(() => new AlpacaTradingService(
    ALPACA_CONFIG.apiKey,
    ALPACA_CONFIG.secretKey,
    ALPACA_CONFIG.isLiveTrading
  ));

  const [alerts, setAlerts] = useState<TradingAlert[]>([]);
  const [systemStatus, setSystemStatus] = useState<AgentStatus[]>([]);
  const [activeTab, setActiveTab] = useState<'discovery' | 'portfolio' | 'alerts' | 'system'>('discovery');
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // Test connections on mount
  useEffect(() => {
    const testConnections = async () => {
      try {
        // Test discovery system connection
        const discoveryStatus = await DiscoveryService.testConnection();

        // Test Alpaca connection
        let alpacaStatus = false;
        try {
          await alpacaService.getAccountInfo();
          alpacaStatus = true;
        } catch (error) {
          console.warn('Alpaca connection failed:', error);
        }

        setIsConnected(discoveryStatus && alpacaStatus);

        if (!discoveryStatus) {
          addAlert({
            type: 'system',
            severity: 'high',
            title: 'Discovery System Offline',
            message: 'Unable to connect to the stock discovery system. Please check the backend service.',
            actionRequired: true
          });
        }

        if (!alpacaStatus) {
          addAlert({
            type: 'system',
            severity: 'medium',
            title: 'Alpaca Connection Issue',
            message: 'Unable to connect to Alpaca. Trading functionality may be limited.',
            actionRequired: true
          });
        }

      } catch (error) {
        console.error('Connection test failed:', error);
        setIsConnected(false);
      }
    };

    testConnections();

    // Set up periodic connection checks
    const connectionCheckInterval = setInterval(testConnections, 60000); // Check every minute

    return () => clearInterval(connectionCheckInterval);
  }, [alpacaService]);

  // Set up WebSocket connection for real-time updates
  useEffect(() => {
    const setupWebSocket = () => {
      const ws = DiscoveryService.createWebSocketConnection(
        (stockUpdate) => {
          // Handle individual stock updates
          console.log('Stock update received:', stockUpdate);
          queryClient.invalidateQueries('explosive-stocks');
        },
        (marketUpdate) => {
          // Handle market metrics updates
          console.log('Market update received:', marketUpdate);
          queryClient.invalidateQueries('market-metrics');
        },
        (error) => {
          console.error('WebSocket error:', error);
          addAlert({
            type: 'system',
            severity: 'medium',
            title: 'Real-time Connection Lost',
            message: 'Lost connection to real-time updates. Data may be delayed.',
            actionRequired: false
          });
        }
      );

      return ws;
    };

    const ws = setupWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Helper function to add alerts
  const addAlert = (alertData: Omit<TradingAlert, 'id' | 'timestamp' | 'acknowledged'>) => {
    const newAlert: TradingAlert = {
      ...alertData,
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date(),
      acknowledged: false
    };

    setAlerts(prev => [newAlert, ...prev.slice(0, 9)]); // Keep last 10 alerts
  };

  // Handle alert acknowledgment
  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-900 text-white">
        {/* Header */}
        <header className="bg-black/50 border-b border-gray-700 px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                🚀 Daily Trading System
              </h1>
              <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                isConnected
                  ? 'bg-green-500/20 text-green-400 border border-green-500'
                  : 'bg-red-500/20 text-red-400 border border-red-500'
              }`}>
                {isConnected ? '● LIVE' : '● OFFLINE'}
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Trading Mode Indicator */}
              <div className={`px-3 py-1 rounded text-xs font-bold ${
                ALPACA_CONFIG.isLiveTrading
                  ? 'bg-red-600 text-white'
                  : 'bg-blue-600 text-white'
              }`}>
                {ALPACA_CONFIG.isLiveTrading ? '🔴 LIVE TRADING' : '📄 PAPER TRADING'}
              </div>

              {/* Alert indicator */}
              {alerts.filter(a => !a.acknowledged).length > 0 && (
                <div className="bg-red-500 text-white px-2 py-1 rounded-full text-xs font-bold">
                  {alerts.filter(a => !a.acknowledged).length}
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Navigation Tabs */}
        <nav className="bg-gray-800 border-b border-gray-700 px-6">
          <div className="flex space-x-8">
            {[
              { key: 'discovery', label: '🔥 Stock Discovery', badge: null },
              { key: 'portfolio', label: '💼 Portfolio', badge: null },
              { key: 'alerts', label: '🚨 Alerts', badge: alerts.filter(a => !a.acknowledged).length || null },
              { key: 'system', label: '⚙️ System', badge: null }
            ].map(({ key, label, badge }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === key
                    ? 'border-green-500 text-green-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                }`}
              >
                {label}
                {badge && (
                  <span className="ml-2 bg-red-500 text-white px-2 py-1 rounded-full text-xs">
                    {badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </nav>

        {/* Main Content */}
        <main className="p-6">
          {activeTab === 'discovery' && (
            <ExplosiveStockDiscovery
              alpacaService={alpacaService}
            />
          )}

          {activeTab === 'portfolio' && (
            <PortfolioOverview
              alpacaService={alpacaService}
            />
          )}

          {activeTab === 'alerts' && (
            <AlertsPanel
              alerts={alerts}
              onAcknowledgeAlert={acknowledgeAlert}
            />
          )}

          {activeTab === 'system' && (
            <SystemMetrics
              systemStatus={systemStatus}
              isConnected={isConnected}
            />
          )}
        </main>

        {/* Quick Stats Footer */}
        <footer className="bg-black/30 border-t border-gray-700 px-6 py-3">
          <div className="flex justify-between items-center text-sm text-gray-400">
            <div className="flex space-x-6">
              <span>🎯 Target: +63.8% Annual Returns</span>
              <span>📊 Multi-Agent System Active</span>
              <span>⚡ Real-time Discovery Engine</span>
            </div>
            <div className="flex space-x-4">
              <span>Last Update: {new Date().toLocaleTimeString()}</span>
              <span>Market: {new Date().getHours() >= 9 && new Date().getHours() < 16 ? 'OPEN' : 'CLOSED'}</span>
            </div>
          </div>
        </footer>
      </div>
    </QueryClientProvider>
  );
};

export default TradingDashboard;