import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { io, Socket } from 'socket.io-client';
import { AgentStatus, SystemMetrics, Alert } from '../types';

interface AgentStatusState {
  agents: Record<string, AgentStatus>;
  systemMetrics: SystemMetrics | null;
  alerts: Alert[];
  isConnected: boolean;
}

type Action = 
  | { type: 'UPDATE_AGENT'; payload: AgentStatus }
  | { type: 'UPDATE_SYSTEM_METRICS'; payload: SystemMetrics }
  | { type: 'ADD_ALERT'; payload: Alert }
  | { type: 'REMOVE_ALERT'; payload: string }
  | { type: 'SET_CONNECTION_STATUS'; payload: boolean };

const initialState: AgentStatusState = {
  agents: {},
  systemMetrics: null,
  alerts: [],
  isConnected: false,
};

const agentStatusReducer = (state: AgentStatusState, action: Action): AgentStatusState => {
  switch (action.type) {
    case 'UPDATE_AGENT':
      return {
        ...state,
        agents: {
          ...state.agents,
          [action.payload.name]: action.payload,
        },
      };
    case 'UPDATE_SYSTEM_METRICS':
      return {
        ...state,
        systemMetrics: action.payload,
      };
    case 'ADD_ALERT':
      return {
        ...state,
        alerts: [action.payload, ...state.alerts.slice(0, 49)], // Keep last 50 alerts
      };
    case 'REMOVE_ALERT':
      return {
        ...state,
        alerts: state.alerts.filter(alert => alert.id !== action.payload),
      };
    case 'SET_CONNECTION_STATUS':
      return {
        ...state,
        isConnected: action.payload,
      };
    default:
      return state;
  }
};

const AgentStatusContext = createContext<{
  state: AgentStatusState;
  dispatch: React.Dispatch<Action>;
  socket: Socket | null;
} | null>(null);

export const AgentStatusProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(agentStatusReducer, initialState);
  const [socket, setSocket] = React.useState<Socket | null>(null);

  useEffect(() => {
    const newSocket = io(process.env.REACT_APP_WEBSOCKET_URL || 'http://localhost:3001', {
      transports: ['websocket'],
    });

    newSocket.on('connect', () => {
      console.log('Connected to agent orchestration system');
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: true });
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from agent orchestration system');
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: false });
    });

    newSocket.on('agent_status', (agentStatus: AgentStatus) => {
      dispatch({ type: 'UPDATE_AGENT', payload: agentStatus });
    });

    newSocket.on('system_metrics', (metrics: SystemMetrics) => {
      dispatch({ type: 'UPDATE_SYSTEM_METRICS', payload: metrics });
    });

    newSocket.on('alert', (alert: Alert) => {
      dispatch({ type: 'ADD_ALERT', payload: alert });
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  return (
    <AgentStatusContext.Provider value={{ state, dispatch, socket }}>
      {children}
    </AgentStatusContext.Provider>
  );
};

export const useAgentStatus = () => {
  const context = useContext(AgentStatusContext);
  if (!context) {
    throw new Error('useAgentStatus must be used within an AgentStatusProvider');
  }
  return context;
};