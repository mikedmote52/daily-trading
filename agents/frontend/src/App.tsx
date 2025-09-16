import React from 'react';
import { TradingDashboard } from './components/TradingDashboard';
import { AgentStatusProvider } from './contexts/AgentStatusContext';
import { TradingProvider } from './contexts/TradingContext';
import './App.css';

function App() {
  return (
    <AgentStatusProvider>
      <TradingProvider>
        <div className="App">
          <TradingDashboard />
        </div>
      </TradingProvider>
    </AgentStatusProvider>
  );
}

export default App;