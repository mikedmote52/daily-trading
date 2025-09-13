# Daily Trading System

AI-powered multi-agent stock trading system with specialized agents for discovery, backtesting, and portfolio management.

## Architecture

### Agent Structure
- **Master Agent**: Orchestrates all other agents and system coordination
- **Frontend Agent**: React/TypeScript web interface for visualizations and controls
- **Backend Agent**: API coordination and data flow management
- **Discovery Agent**: Stock screening and selection using AI-driven criteria
- **Backtesting Agent**: Historical strategy testing and optimization
- **Portfolio Agent**: Position management and execution via Alpaca API

### Directory Structure
```
daily-trading/
├── agents/
│   ├── master/          # Master orchestration agent
│   ├── frontend/        # React web interface
│   ├── backend/         # API coordination hub
│   ├── discovery/       # Stock discovery and screening
│   ├── backtesting/     # Strategy testing and optimization
│   └── portfolio/       # Portfolio management and execution
├── shared/
│   ├── utils/           # Shared utilities
│   ├── types/           # TypeScript type definitions
│   └── config/          # Configuration files
└── data/
    ├── historical/      # Historical market data
    ├── realtime/        # Real-time data cache
    └── models/          # AI models and weights
```

## Quick Start

1. Install dependencies:
   ```bash
   npm run install:all
   ```

2. Set up environment variables (see `.env.example`)

3. Start all agents:
   ```bash
   npm run dev
   ```

## Environment Setup

Required API keys and configurations:
- Alpaca Trading API
- Financial data providers
- Claude API keys for each agent

See individual agent directories for specific setup instructions.