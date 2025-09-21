import React from 'react';
import { AlertTriangle, Info, CheckCircle, XCircle, Clock, Bell } from 'lucide-react';
import { TradingAlert } from '../types/trading';

interface AlertsPanelProps {
  alerts: TradingAlert[];
  onAcknowledgeAlert: (alertId: string) => void;
}

export const AlertsPanel: React.FC<AlertsPanelProps> = ({ alerts, onAcknowledgeAlert }) => {

  const getAlertIcon = (type: string, severity: string) => {
    if (severity === 'critical') {
      return <XCircle className="h-5 w-5 text-red-500" />;
    } else if (severity === 'high') {
      return <AlertTriangle className="h-5 w-5 text-red-400" />;
    } else if (severity === 'medium') {
      return <Clock className="h-5 w-5 text-yellow-400" />;
    } else {
      return <CheckCircle className="h-5 w-5 text-green-400" />;
    }
  };

  const getAlertBorderColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'border-green-400';
      case 'error':
        return 'border-red-400';
      case 'warning':
        return 'border-yellow-400';
      case 'info':
      default:
        return 'border-blue-400';
    }
  };

  const getAlertBgColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'bg-green-900/20';
      case 'error':
        return 'bg-red-900/20';
      case 'warning':
        return 'bg-yellow-900/20';
      case 'info':
      default:
        return 'bg-blue-900/20';
    }
  };

  const dismissAlert = (id: string) => {
    onAcknowledgeAlert(id);
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const alertsByType = alerts.reduce((acc, alert) => {
    acc[alert.type] = (acc[alert.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="space-y-6">
      {/* Alert Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Total Alerts</p>
              <p className="text-2xl font-bold text-white">{alerts.length}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Errors</p>
              <p className="text-2xl font-bold text-red-400">{alertsByType.error || 0}</p>
            </div>
            <XCircle className="h-8 w-8 text-red-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Warnings</p>
              <p className="text-2xl font-bold text-yellow-400">{alertsByType.warning || 0}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-yellow-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Success</p>
              <p className="text-2xl font-bold text-green-400">{alertsByType.success || 0}</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </div>
      </div>

      {/* Alerts List */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">System Alerts</h3>
          {alerts.length > 0 && (
            <button
              onClick={() => alerts.forEach(alert => dismissAlert(alert.id))}
              className="px-3 py-1 text-sm bg-gray-600 hover:bg-gray-500 rounded text-white transition-colors"
            >
              Clear All
            </button>
          )}
        </div>

        {alerts.length > 0 ? (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`border-l-4 ${getAlertBorderColor(alert.type)} ${getAlertBgColor(alert.type)} rounded-r-lg p-4`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3 flex-1">
                    <div className="flex-shrink-0 pt-0.5">
                      {getAlertIcon(alert.type, alert.severity || 'low')}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        {(alert as any).agent && (
                          <span className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded-full">
                            {(alert as any).agent}
                          </span>
                        )}
                        <span className="text-xs text-gray-400 flex items-center">
                          <Clock className="h-3 w-3 mr-1" />
                          {formatTime(alert.timestamp.toString())}
                        </span>
                      </div>
                      <p className="text-sm text-white leading-relaxed">{alert.message}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => dismissAlert(alert.id)}
                    className="ml-4 flex-shrink-0 text-gray-400 hover:text-gray-200 transition-colors"
                  >
                    <XCircle className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="text-gray-400">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">No alerts</p>
              <p className="text-sm">Your system is running smoothly</p>
            </div>
          </div>
        )}
      </div>

      {/* Recent Activity Timeline */}
      {alerts.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-6">Recent Activity Timeline</h3>
          <div className="space-y-4">
            {alerts.slice(0, 10).map((alert, index) => (
              <div key={alert.id} className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getAlertBgColor(alert.type)}`}>
                    {getAlertIcon(alert.type, alert.severity || 'low')}
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <p className="text-sm text-white truncate">{alert.message}</p>
                    {(alert as any).agent && (
                      <span className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
                        {(alert as any).agent}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mt-1">{formatTime(alert.timestamp.toString())}</p>
                </div>
                {index < alerts.length - 1 && (
                  <div className="absolute left-4 mt-8 w-0.5 h-4 bg-gray-600" />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};