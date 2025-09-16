import React from 'react';
import { AlertTriangle, CheckCircle, XCircle, Clock, Bell } from 'lucide-react';
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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'border-red-600 bg-red-900/30 shadow-red-500/20';
      case 'high':
        return 'border-red-500 bg-red-900/20 shadow-red-500/10';
      case 'medium':
        return 'border-yellow-500 bg-yellow-900/20 shadow-yellow-500/10';
      case 'low':
        return 'border-green-500 bg-green-900/20 shadow-green-500/10';
      default:
        return 'border-gray-500 bg-gray-900/20';
    }
  };

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'opportunity':
        return '🎯 Opportunity';
      case 'risk':
        return '⚠️ Risk Alert';
      case 'execution':
        return '💼 Execution';
      case 'system':
        return '⚙️ System';
      default:
        return '📢 Alert';
    }
  };

  const unacknowledgedAlerts = alerts.filter(alert => !alert.acknowledged);
  const acknowledgedAlerts = alerts.filter(alert => alert.acknowledged);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-900 to-red-900 rounded-lg p-6 border border-orange-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Bell className="h-8 w-8 text-orange-400" />
            <div>
              <h2 className="text-3xl font-bold text-white">🚨 Alerts & Notifications</h2>
              <p className="text-orange-200">System alerts and trading notifications</p>
            </div>
          </div>
          <div className="text-right">
            {unacknowledgedAlerts.length > 0 && (
              <div className="bg-red-600 text-white px-4 py-2 rounded-full font-bold">
                {unacknowledgedAlerts.length} Unread
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Unacknowledged Alerts */}
      {unacknowledgedAlerts.length > 0 && (
        <div>
          <h3 className="text-xl font-bold text-red-400 mb-4">🔔 Active Alerts</h3>
          <div className="space-y-3">
            {unacknowledgedAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`border rounded-lg p-4 shadow-lg border-l-4 ${getSeverityColor(alert.severity)}`}
              >
                <div className="flex items-start space-x-3">
                  {getAlertIcon(alert.type, alert.severity)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <h3 className="text-lg font-semibold text-white">{alert.title}</h3>
                        <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
                          {getTypeLabel(alert.type)}
                        </span>
                      </div>
                      <span className="text-xs text-gray-400">
                        {alert.timestamp.toLocaleTimeString()}
                      </span>
                    </div>

                    <p className="text-gray-300 mb-3">{alert.message}</p>

                    {alert.symbol && (
                      <div className="bg-black/30 rounded px-2 py-1 inline-block mb-3">
                        <span className="text-blue-400 font-bold">{alert.symbol}</span>
                      </div>
                    )}

                    <div className="flex items-center justify-between">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => onAcknowledgeAlert(alert.id)}
                          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm transition-colors"
                        >
                          ✓ Acknowledge
                        </button>
                        {alert.actionRequired && (
                          <span className="bg-yellow-600 text-white px-3 py-2 rounded text-sm">
                            Action Required
                          </span>
                        )}
                      </div>
                      <div className={`text-xs font-bold ${
                        alert.severity === 'critical' ? 'text-red-400' :
                        alert.severity === 'high' ? 'text-red-300' :
                        alert.severity === 'medium' ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {alert.severity.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Acknowledged Alerts */}
      {acknowledgedAlerts.length > 0 && (
        <div>
          <h3 className="text-xl font-bold text-gray-400 mb-4">📋 Recent Activity</h3>
          <div className="space-y-2">
            {acknowledgedAlerts.slice(0, 5).map((alert) => (
              <div
                key={alert.id}
                className="border border-gray-600 rounded-lg p-3 bg-gray-800/50 opacity-70"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getAlertIcon(alert.type, alert.severity)}
                    <div>
                      <h4 className="text-sm font-medium text-gray-300">{alert.title}</h4>
                      <p className="text-xs text-gray-400">{alert.message}</p>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    {alert.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Alerts */}
      {alerts.length === 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-12 text-center">
          <CheckCircle className="h-16 w-16 text-green-400 mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-white mb-2">All Clear! 🎉</h3>
          <p className="text-gray-400">No active alerts or notifications</p>
          <p className="text-sm text-gray-500 mt-2">
            System is running smoothly and monitoring for opportunities
          </p>
        </div>
      )}
    </div>
  );
};