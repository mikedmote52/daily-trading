import React, { useState } from 'react';
import { AgentStatusPanel } from './AgentStatusPanel';
import { PortfolioOverview } from './PortfolioOverview';
import { WatchlistPanel } from './WatchlistPanel';
import { TradeExecutionPanel } from './TradeExecutionPanel';
import { BacktestingPanel } from './BacktestingPanel';
import { AlertsPanel } from './AlertsPanel';
import { SystemMetrics } from './SystemMetrics';

type TabType = 'overview' | 'watchlist' | 'trading' | 'backtesting' | 'agents' | 'alerts';

export const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  const tabs: { id: TabType; label: string }[] = [
    { id: 'overview', label: 'Portfolio' },
    { id: 'watchlist', label: 'Watchlist' },
    { id: 'trading', label: 'Trading' },
    { id: 'backtesting', label: 'Backtesting' },
    { id: 'agents', label: 'Agents' },
    { id: 'alerts', label: 'Alerts' }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-white">Daily Trading System</h1>
            </div>
            <div className="flex items-center space-x-4">
              <SystemMetrics />
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <PortfolioOverview />
            </div>
          )}
          
          {activeTab === 'watchlist' && (
            <div className="space-y-6">
              <WatchlistPanel />
            </div>
          )}
          
          {activeTab === 'trading' && (
            <div className="space-y-6">
              <TradeExecutionPanel />
            </div>
          )}
          
          {activeTab === 'backtesting' && (
            <div className="space-y-6">
              <BacktestingPanel />
            </div>
          )}
          
          {activeTab === 'agents' && (
            <div className="space-y-6">
              <AgentStatusPanel />
            </div>
          )}
          
          {activeTab === 'alerts' && (
            <div className="space-y-6">
              <AlertsPanel alerts={[]} onAcknowledgeAlert={() => {}} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
};