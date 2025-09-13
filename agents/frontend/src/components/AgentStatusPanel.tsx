import React from 'react';
import { Activity, Clock, Cpu, AlertTriangle } from 'lucide-react';
import { useAgentStatus } from '../contexts/AgentStatusContext';
import { AgentStatus } from '../types';

const AgentCard: React.FC<{ agent: AgentStatus }> = ({ agent }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'bg-green-500';
      case 'busy':
        return 'bg-yellow-500';
      case 'offline':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <Activity className="h-4 w-4" />;
      case 'busy':
        return <Cpu className="h-4 w-4" />;
      case 'offline':
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`} />
          <h3 className="text-lg font-semibold text-white capitalize">
            {agent.name.replace(/([A-Z])/g, ' $1').trim()} Agent
          </h3>
        </div>
        <div className="flex items-center space-x-2 text-gray-400">
          {getStatusIcon(agent.status)}
          <span className="text-sm capitalize">{agent.status}</span>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <span className="text-sm text-gray-400">Last Heartbeat:</span>
          <span className="ml-2 text-sm text-white">
            {formatTime(agent.lastHeartbeat)}
          </span>
        </div>

        {agent.currentTask && (
          <div>
            <span className="text-sm text-gray-400">Current Task:</span>
            <span className="ml-2 text-sm text-white">{agent.currentTask}</span>
          </div>
        )}

        {agent.performanceMetrics && Object.keys(agent.performanceMetrics).length > 0 && (
          <div>
            <span className="text-sm text-gray-400 block mb-2">Performance Metrics:</span>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(agent.performanceMetrics).map(([key, value]) => (
                <div key={key} className="bg-gray-700 rounded p-2">
                  <div className="text-xs text-gray-400 uppercase">{key}</div>
                  <div className="text-sm text-white font-medium">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export const AgentStatusPanel: React.FC = () => {
  const { state } = useAgentStatus();
  const { agents } = state;

  const agentList = Object.values(agents);
  const onlineCount = agentList.filter(a => a.status === 'online').length;
  const busyCount = agentList.filter(a => a.status === 'busy').length;
  const offlineCount = agentList.filter(a => a.status === 'offline').length;

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Total Agents</p>
              <p className="text-2xl font-bold text-white">{agentList.length}</p>
            </div>
            <Cpu className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Online</p>
              <p className="text-2xl font-bold text-green-400">{onlineCount}</p>
            </div>
            <Activity className="h-8 w-8 text-green-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Busy</p>
              <p className="text-2xl font-bold text-yellow-400">{busyCount}</p>
            </div>
            <Clock className="h-8 w-8 text-yellow-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Offline</p>
              <p className="text-2xl font-bold text-red-400">{offlineCount}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-red-400" />
          </div>
        </div>
      </div>

      {/* Agent Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agentList.map((agent) => (
          <AgentCard key={agent.name} agent={agent} />
        ))}
      </div>
    </div>
  );
};