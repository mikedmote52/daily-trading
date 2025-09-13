import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Dashboard } from './components/Dashboard';
import { AgentStatusProvider } from './contexts/AgentStatusContext';
import { TradingProvider } from './contexts/TradingContext';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Refetch every 5 seconds
      refetchIntervalInBackground: true,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AgentStatusProvider>
        <TradingProvider>
          <div className="App min-h-screen bg-gray-900 text-white">
            <Dashboard />
          </div>
        </TradingProvider>
      </AgentStatusProvider>
    </QueryClientProvider>
  );
}

export default App;