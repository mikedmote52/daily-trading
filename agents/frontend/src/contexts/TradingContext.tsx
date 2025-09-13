import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { useQuery } from 'react-query';
import axios from 'axios';
import { Stock, Position, Trade, BacktestResult } from '../types';

interface TradingState {
  watchlist: Stock[];
  positions: Position[];
  trades: Trade[];
  backtestResults: BacktestResult[];
  portfolioValue: number;
  dailyPnl: number;
  isLoading: boolean;
  error: string | null;
}

type TradingAction =
  | { type: 'SET_WATCHLIST'; payload: Stock[] }
  | { type: 'SET_POSITIONS'; payload: Position[] }
  | { type: 'SET_TRADES'; payload: Trade[] }
  | { type: 'ADD_TRADE'; payload: Trade }
  | { type: 'SET_BACKTEST_RESULTS'; payload: BacktestResult[] }
  | { type: 'UPDATE_PORTFOLIO_VALUE'; payload: number }
  | { type: 'UPDATE_DAILY_PNL'; payload: number }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null };

const initialState: TradingState = {
  watchlist: [],
  positions: [],
  trades: [],
  backtestResults: [],
  portfolioValue: 0,
  dailyPnl: 0,
  isLoading: false,
  error: null,
};

const tradingReducer = (state: TradingState, action: TradingAction): TradingState => {
  switch (action.type) {
    case 'SET_WATCHLIST':
      return { ...state, watchlist: action.payload };
    case 'SET_POSITIONS':
      return { ...state, positions: action.payload };
    case 'SET_TRADES':
      return { ...state, trades: action.payload };
    case 'ADD_TRADE':
      return { ...state, trades: [action.payload, ...state.trades] };
    case 'SET_BACKTEST_RESULTS':
      return { ...state, backtestResults: action.payload };
    case 'UPDATE_PORTFOLIO_VALUE':
      return { ...state, portfolioValue: action.payload };
    case 'UPDATE_DAILY_PNL':
      return { ...state, dailyPnl: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    default:
      return state;
  }
};

const TradingContext = createContext<{
  state: TradingState;
  dispatch: React.Dispatch<TradingAction>;
  executeTradeCommand: (command: string) => Promise<void>;
  runBacktest: (strategy: string, parameters: Record<string, any>) => Promise<void>;
} | null>(null);

export const TradingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(tradingReducer, initialState);

  // Fetch watchlist data
  const { data: watchlistData } = useQuery(
    'watchlist',
    () => axios.get('/api/watchlist').then(res => res.data),
    {
      onSuccess: (data) => {
        dispatch({ type: 'SET_WATCHLIST', payload: data });
      },
      onError: (error: any) => {
        dispatch({ type: 'SET_ERROR', payload: error.message });
      }
    }
  );

  // Fetch positions data
  const { data: positionsData } = useQuery(
    'positions',
    () => axios.get('/api/positions').then(res => res.data),
    {
      onSuccess: (data) => {
        dispatch({ type: 'SET_POSITIONS', payload: data.positions || [] });
        dispatch({ type: 'UPDATE_PORTFOLIO_VALUE', payload: data.totalValue || 0 });
        dispatch({ type: 'UPDATE_DAILY_PNL', payload: data.dailyPnl || 0 });
      },
      onError: (error: any) => {
        dispatch({ type: 'SET_ERROR', payload: error.message });
      }
    }
  );

  // Fetch trades data
  const { data: tradesData } = useQuery(
    'trades',
    () => axios.get('/api/trades').then(res => res.data),
    {
      onSuccess: (data) => {
        dispatch({ type: 'SET_TRADES', payload: data });
      },
      onError: (error: any) => {
        dispatch({ type: 'SET_ERROR', payload: error.message });
      }
    }
  );

  // Fetch backtest results
  const { data: backtestData } = useQuery(
    'backtest-results',
    () => axios.get('/api/backtest/results').then(res => res.data),
    {
      onSuccess: (data) => {
        dispatch({ type: 'SET_BACKTEST_RESULTS', payload: data });
      },
      onError: (error: any) => {
        dispatch({ type: 'SET_ERROR', payload: error.message });
      }
    }
  );

  const executeTradeCommand = async (command: string): Promise<void> => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      const response = await axios.post('/api/trades/execute', { command });
      
      if (response.data.trade) {
        dispatch({ type: 'ADD_TRADE', payload: response.data.trade });
      }
      
      dispatch({ type: 'SET_ERROR', payload: null });
    } catch (error: any) {
      dispatch({ type: 'SET_ERROR', payload: error.response?.data?.message || error.message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const runBacktest = async (strategy: string, parameters: Record<string, any>): Promise<void> => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      await axios.post('/api/backtest/run', { strategy, parameters });
      
      // Results will be updated via the backtest-results query refetch
      dispatch({ type: 'SET_ERROR', payload: null });
    } catch (error: any) {
      dispatch({ type: 'SET_ERROR', payload: error.response?.data?.message || error.message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  return (
    <TradingContext.Provider value={{ 
      state, 
      dispatch, 
      executeTradeCommand, 
      runBacktest 
    }}>
      {children}
    </TradingContext.Provider>
  );
};

export const useTrading = () => {
  const context = useContext(TradingContext);
  if (!context) {
    throw new Error('useTrading must be used within a TradingProvider');
  }
  return context;
};