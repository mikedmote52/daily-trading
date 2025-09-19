# 🔑 Fix Alpaca Trading Authentication

## The Problem
You're getting "Alpaca API error: unauthorized" because the trading service needs proper API credentials.

## Quick Fix (5 minutes)

### Step 1: Get Your Alpaca Paper Trading Keys
1. Go to https://app.alpaca.markets/
2. Sign in to your Alpaca account
3. Navigate to **Paper Trading** section
4. Go to **API Keys**
5. Copy your:
   - **API Key ID** (starts with PK...)
   - **Secret Key** (starts with ...)

### Step 2: Configure Render Service
1. Go to https://dashboard.render.com/
2. Find your **alphastack-orders** service
3. Click **Environment** tab
4. Add these environment variables:

```bash
ALPACA_KEY=PK... (your API Key ID)
ALPACA_SECRET=... (your Secret Key)
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Step 3: Test the Fix
1. Wait 2-3 minutes for service to restart
2. Go to https://alphastack-frontend.onrender.com/
3. Click **🔢 Test Trading** button
4. Should show "✅ Trading Connection Successful!"

### Step 4: Try Buying Again
1. Click any **BUY** button on a stock
2. Enter position size (e.g., $100)
3. Click **Buy X Shares**
4. Should show success message instead of error

## If Still Not Working

### Check Orders Service Health
Visit: https://alphastack-orders.onrender.com/health

Should show:
```json
{
  "status": "healthy",
  "alpaca_configured": true
}
```

If `alpaca_configured` is `false`, the API keys aren't properly set.

### Alternative: Create New Alpaca Keys
1. In Alpaca dashboard, **delete** existing API keys
2. **Create new** paper trading API keys
3. Update Render environment variables with new keys
4. Wait for service restart

## Troubleshooting

### Error: "API key not found"
- API Key ID is wrong or missing
- Check you're using **Paper Trading** keys, not Live Trading

### Error: "Invalid secret"
- Secret key is wrong or missing
- Make sure no extra spaces in the environment variables

### Error: "Account not found"
- Using Live Trading URL with Paper Trading keys
- Ensure `ALPACA_BASE_URL=https://paper-api.alpaca.markets`

## Expected Behavior After Fix

### ✅ Working System
- Test Trading button shows account balance
- Buy orders complete successfully
- Orders appear in your Alpaca paper trading account
- No "unauthorized" errors

### 📊 Trading Features
- **Paper Trading**: Safe virtual money trading
- **Bracket Orders**: Automatic stop-loss and take-profit
- **Risk Management**: Built-in position sizing
- **Real Market Data**: Live prices from exchanges

## Security Notes
- These are **Paper Trading** keys (virtual money only)
- Keys are stored securely in Render environment
- Never share API keys publicly
- Can regenerate keys anytime in Alpaca dashboard

Once configured, you'll be able to test explosive stock strategies risk-free with virtual money while using real market data and order execution systems!
