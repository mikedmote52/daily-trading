import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import PortfolioApp from "./portfolio.jsx";

const discoveryBase = import.meta.env.VITE_DISCOVERY_API_URL;
const ordersBase    = import.meta.env.VITE_ORDERS_API_URL;
const portfolioBase = import.meta.env.VITE_PORTFOLIO_API_URL;

function useHealth(url) {
  const [state, setState] = useState({ status: "checking", code: null, details: null });
  useEffect(() => {
    if (!url) { setState({ status: "missing", code: null, details: null }); return; }
    fetch(`${url.replace(/\/$/, "")}/health`)
      .then(async r => {
        const details = r.ok ? await r.json().catch(() => null) : null;
        setState({
          status: r.ok ? "ok" : "error",
          code: r.status,
          details: details
        });
      })
      .catch(() => setState({ status: "error", code: null, details: null }));
  }, [url]);
  return state;
}

function Pill({ ok, label, code }) {
  const bg = ok ? "#16a34a" : "#dc2626";
  return (
    <span style={{
      background: bg, color: "white", padding: "4px 10px",
      borderRadius: 999, fontSize: 12, marginLeft: 8
    }}>
      {ok ? "OK" : "FAIL"}{code ? ` • ${code}` : ""}
    </span>
  );
}

function useStocks() {
  const [stocks, setStocks] = useState({ data: [], loading: true, error: null });

  useEffect(() => {
    if (!discoveryBase) {
      setStocks({ data: [], loading: false, error: "No discovery API URL" });
      return;
    }

    fetch(`${discoveryBase.replace(/\/$/, "")}/signals/top`)
      .then(r => r.json())
      .then(data => {
        setStocks({
          data: data.signals || data.final_recommendations || [],
          loading: false,
          error: null
        });
      })
      .catch(err => {
        setStocks({ data: [], loading: false, error: err.message });
      });
  }, []);

  return stocks;
}

function BuyButton({ symbol, price, stopLoss, priceTarget }) {
  const [showModal, setShowModal] = useState(false);
  const [positionSize, setPositionSize] = useState(1000);
  const [buying, setBuying] = useState(false);

  const shares = Math.floor(positionSize / price);
  const actualInvestment = shares * price;
  const stopLossPrice = stopLoss || (price * 0.9);  // 10% stop if not provided
  const profitTarget = priceTarget || (price * 1.638);  // 63.8% target if not provided

  const maxLoss = actualInvestment - (shares * stopLossPrice);
  const maxProfit = (shares * profitTarget) - actualInvestment;
  const riskReward = maxProfit / maxLoss;

  const handleBuy = async () => {
    if (!ordersBase) {
      alert("❌ Orders API not configured\n\nPlease contact support to enable trading functionality.");
      return;
    }

    // Validation checks
    if (shares <= 0) {
      alert("❌ Invalid position size\n\nPlease enter a valid dollar amount.");
      return;
    }

    if (price <= 0) {
      alert("❌ Invalid price data\n\nUnable to get current price for this stock.");
      return;
    }

    setBuying(true);
    try {
      const idempotencyKey = `order-${symbol}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      const orderPayload = {
        ticker: symbol,
        notional_usd: actualInvestment,
        last_price: price,
        side: 'buy'
      };

      console.log('Submitting order:', orderPayload);

      const response = await fetch(`${ordersBase.replace(/\/$/, "")}/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': idempotencyKey
        },
        body: JSON.stringify(orderPayload)
      });

      const responseText = await response.text();
      console.log('Order response:', response.status, responseText);

      if (response.ok) {
        let result;
        try {
          result = JSON.parse(responseText);
        } catch {
          result = { message: responseText };
        }

        alert(`✅ Buy Order Submitted Successfully!\n\n` +
              `📊 Stock: ${symbol}\n` +
              `📈 Shares: ${shares}\n` +
              `💰 Total Investment: $${actualInvestment.toFixed(2)}\n` +
              `🛑 Stop Loss: $${stopLossPrice.toFixed(2)} (-${((1 - stopLossPrice/price) * 100).toFixed(1)}%)\n` +
              `🎯 Profit Target: $${profitTarget.toFixed(2)} (+${((profitTarget/price - 1) * 100).toFixed(1)}%)\n` +
              `📊 Risk/Reward: 1:${riskReward.toFixed(1)}\n\n` +
              `✨ Order will execute at market open if after hours.`);
        setShowModal(false);
      } else {
        let errorMessage;
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.detail || errorData.message || responseText;
        } catch {
          errorMessage = responseText;
        }

        // Enhanced error handling
        if (response.status === 401 || errorMessage.includes('unauthorized')) {
          alert(`🔐 Authentication Error\n\n` +
                `The trading system is not properly authenticated with Alpaca.\n\n` +
                `📧 Please contact support to configure your trading account.\n\n` +
                `💡 This is likely due to missing or invalid API credentials.`);
        } else if (response.status === 400) {
          alert(`⚠️ Invalid Order Request\n\n` +
                `There was an issue with your order parameters:\n\n` +
                `${errorMessage}\n\n` +
                `💡 Try adjusting your position size or check if the market is open.`);
        } else if (response.status === 403) {
          alert(`🚫 Trading Restricted\n\n` +
                `Your account may have trading restrictions.\n\n` +
                `📧 Please contact support for assistance.`);
        } else if (response.status >= 500) {
          alert(`⚡ Server Error\n\n` +
                `The trading system is temporarily unavailable.\n\n` +
                `🔄 Please try again in a few moments.\n\n` +
                `Error: ${errorMessage}`);
        } else {
          alert(`❌ Order Failed\n\n` +
                `Status: ${response.status}\n` +
                `Error: ${errorMessage}\n\n` +
                `💡 Please try again or contact support if the issue persists.`);
        }
      }
    } catch (err) {
      console.error('Order submission error:', err);
      alert(`🌐 Network Error\n\n` +
            `Unable to connect to the trading system.\n\n` +
            `🔄 Please check your internet connection and try again.\n\n` +
            `Error: ${err.message}`);
    } finally {
      setBuying(false);
    }
  };

  if (showModal) {
    return (
      <div style={{
        position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
        background: "rgba(0,0,0,0.8)", display: "flex",
        alignItems: "center", justifyContent: "center", zIndex: 1000
      }}>
        <div style={{
          background: "white", padding: 24, borderRadius: 8,
          maxWidth: 400, width: "90%"
        }}>
          <h3 style={{ margin: "0 0 16px 0" }}>Buy {symbol} at ${price}</h3>

          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>
              Position Size ($)
            </label>
            <input
              type="number"
              value={positionSize}
              onChange={(e) => setPositionSize(Number(e.target.value))}
              style={{
                width: "100%", padding: 8, border: "1px solid #ddd",
                borderRadius: 4, fontSize: 14
              }}
            />
          </div>

          <div style={{
            background: "#f8fafc", padding: 12, borderRadius: 4,
            marginBottom: 16, fontSize: 13
          }}>
            <div><strong>Shares:</strong> {shares.toLocaleString()}</div>
            <div><strong>Actual Investment:</strong> ${actualInvestment.toFixed(2)}</div>
            <div style={{ color: "#dc2626" }}><strong>Stop Loss:</strong> ${stopLossPrice.toFixed(2)} (Max Loss: ${maxLoss.toFixed(2)})</div>
            <div style={{ color: "#16a34a" }}><strong>Profit Target:</strong> ${profitTarget.toFixed(2)} (Max Profit: ${maxProfit.toFixed(2)})</div>
            <div><strong>Risk:Reward:</strong> 1:{riskReward.toFixed(1)}</div>
          </div>

          <div style={{
            background: "#fffbeb", border: "1px solid #fed7aa", padding: 8, borderRadius: 4,
            marginBottom: 16, fontSize: 12, color: "#92400e"
          }}>
            📈 <strong>Paper Trading:</strong> This is a simulated trade using virtual money.
            Perfect for testing strategies without real financial risk!
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handleBuy}
              disabled={buying || shares === 0 || price <= 0}
              style={{
                flex: 1,
                background: buying ? "#9ca3af" : (shares === 0 || price <= 0) ? "#dc2626" : "#16a34a",
                color: "white", border: "none", padding: "10px 16px",
                borderRadius: 4,
                cursor: (buying || shares === 0 || price <= 0) ? "not-allowed" : "pointer",
                fontSize: 14, fontWeight: "bold"
              }}
            >
              {buying ? "Placing Order..." :
               shares === 0 ? "Invalid Amount" :
               price <= 0 ? "Invalid Price" :
               `Buy ${shares} Shares`}
            </button>
            <button
              onClick={() => setShowModal(false)}
              style={{
                background: "#6b7280", color: "white", border: "none",
                padding: "10px 16px", borderRadius: 4, cursor: "pointer",
                fontSize: 14
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => setShowModal(true)}
      style={{
        background: "#16a34a", color: "white", border: "none",
        padding: "6px 12px", borderRadius: 4, cursor: "pointer",
        fontSize: 12, fontWeight: "bold"
      }}
    >
      BUY
    </button>
  );
}

function RunDiscoveryButton() {
  const [running, setRunning] = useState(false);

  const handleRunDiscovery = async () => {
    if (!discoveryBase) {
      alert("Discovery API not configured");
      return;
    }

    setRunning(true);
    try {
      const response = await fetch(`${discoveryBase.replace(/\/$/, "")}/discover`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Discovery scan started: ${result.scan_id}\nResults will update automatically.`);
        // Refresh the page after a short delay to pick up new results
        setTimeout(() => window.location.reload(), 3000);
      } else {
        alert(`Failed to start discovery scan: ${response.status}`);
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <button
      onClick={handleRunDiscovery}
      disabled={running}
      style={{
        background: running ? "#9ca3af" : "#2563eb",
        color: "white",
        border: "none",
        padding: "8px 16px",
        borderRadius: 4,
        cursor: running ? "not-allowed" : "pointer",
        fontSize: 14,
        fontWeight: "bold",
        marginBottom: 16
      }}
    >
      {running ? "Running Discovery..." : "🔍 Run Discovery Scan"}
    </button>
  );
}

function TestTradingButton() {
  const [testing, setTesting] = useState(false);

  const handleTestTrading = async () => {
    if (!ordersBase) {
      alert("❌ Orders API not configured");
      return;
    }

    setTesting(true);
    try {
      // Test account endpoint
      const response = await fetch(`${ordersBase.replace(/\/$/, "")}/account`);
      const responseText = await response.text();

      if (response.ok) {
        const accountData = JSON.parse(responseText);
        alert(`✅ Trading Connection Successful!\n\n` +
              `💰 Buying Power: $${parseFloat(accountData.buying_power || 0).toLocaleString()}\n` +
              `💵 Cash: $${parseFloat(accountData.cash || 0).toLocaleString()}\n` +
              `📊 Portfolio Value: $${parseFloat(accountData.portfolio_value || 0).toLocaleString()}\n\n` +
              `✨ Your paper trading account is ready!`);
      } else if (response.status === 401) {
        alert(`🔐 Authentication Failed\n\n` +
              `The Alpaca API credentials are not properly configured.\n\n` +
              `📧 Please contact support to set up your trading account.`);
      } else {
        alert(`❌ Trading Connection Failed\n\n` +
              `Status: ${response.status}\n` +
              `Response: ${responseText}\n\n` +
              `📧 Please contact support for assistance.`);
      }
    } catch (err) {
      alert(`🌐 Network Error\n\n` +
            `Unable to connect to trading system.\n\n` +
            `Error: ${err.message}`);
    } finally {
      setTesting(false);
    }
  };

  return (
    <button
      onClick={handleTestTrading}
      disabled={testing}
      style={{
        background: testing ? "#9ca3af" : "#059669",
        color: "white",
        border: "none",
        padding: "8px 16px",
        borderRadius: 4,
        cursor: testing ? "not-allowed" : "pointer",
        fontSize: 14,
        fontWeight: "bold",
        marginBottom: 16,
        marginLeft: 8
      }}
    >
      {testing ? "Testing..." : "📈 Test Trading"}
    </button>
  );
}

function App() {
  const [currentView, setCurrentView] = useState("discovery");
  const d = useHealth(discoveryBase);
  const o = useHealth(ordersBase);
  const p = useHealth(portfolioBase);
  const stocks = useStocks();

  const navButtonStyle = (isActive) => ({
    padding: "12px 20px",
    margin: "0 8px 0 0",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: "600",
    background: isActive ? "#2563eb" : "#f3f4f6",
    color: isActive ? "white" : "#374151",
    transition: "all 0.2s ease"
  });

  if (currentView === "portfolio") {
    return (
      <div style={{ fontFamily: "ui-sans-serif, system-ui" }}>
        {/* Navigation Header */}
        <div style={{ background: "white", borderBottom: "1px solid #e5e7eb", padding: "16px 24px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <h1 style={{ fontSize: 32, margin: 0 }}>AlphaStack</h1>
              <p style={{ color: "#6b7280", margin: "4px 0 0 0" }}>AI-Powered Explosive Stock Returns</p>
            </div>
            <nav style={{ display: "flex" }}>
              <button
                style={navButtonStyle(currentView === "discovery")}
                onClick={() => setCurrentView("discovery")}
              >
                🔍 Discovery
              </button>
              <button
                style={navButtonStyle(currentView === "portfolio")}
                onClick={() => setCurrentView("portfolio")}
              >
                📊 Portfolio
              </button>
            </nav>
          </div>
        </div>
        <PortfolioApp />
      </div>
    );
  }

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui", padding: 24, lineHeight: 1.4 }}>
      {/* Navigation Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 40, margin: 0 }}>AlphaStack UI</h1>
          <p style={{ color: "#555" }}>Stock discovery and trading interface</p>
        </div>
        <nav style={{ display: "flex" }}>
          <button
            style={navButtonStyle(currentView === "discovery")}
            onClick={() => setCurrentView("discovery")}
          >
            🔍 Discovery
          </button>
          <button
            style={navButtonStyle(currentView === "portfolio")}
            onClick={() => setCurrentView("portfolio")}
          >
            📊 Portfolio
          </button>
        </nav>
      </div>

      <div style={{ display: "grid", gap: 12, maxWidth: 720, marginBottom: 24 }}>
        <div>
          <strong>Discovery API</strong>
          <Pill ok={d.status === "ok"} label="Discovery" code={d.code} />
          <div style={{ color: "#666", fontSize: 13 }}>
            URL: {discoveryBase || <em style={{ color: "#dc2626" }}>VITE_DISCOVERY_API_URL not set</em>}
          </div>
        </div>

        <div>
          <strong>Orders API</strong>
          <Pill ok={o.status === "ok"} label="Orders" code={o.code} />
          <div style={{ color: "#666", fontSize: 13 }}>
            URL: {ordersBase || <em style={{ color: "#dc2626" }}>VITE_ORDERS_API_URL not set</em>}
          </div>
          {o.details && (
            <div style={{ fontSize: 11, color: o.details.alpaca_configured ? "#10b981" : "#dc2626", marginTop: 4 }}>
              Alpaca: {o.details.alpaca_configured ? "✅ Configured" : "❌ Not Configured"}
              {!o.details.alpaca_configured && (
                <div style={{ color: "#dc2626", fontWeight: "bold" }}>
                  ⚠️ Trading disabled - API keys missing
                </div>
              )}
            </div>
          )}
        </div>

        <div>
          <strong>Portfolio API</strong>
          <Pill ok={p.status === "ok"} label="Portfolio" code={p.code} />
          <div style={{ color: "#666", fontSize: 13 }}>
            URL: {portfolioBase || <em style={{ color: "#dc2626" }}>VITE_PORTFOLIO_API_URL not set</em>}
          </div>
        </div>
      </div>

      <hr style={{ margin: "24px 0" }} />

      <div>
        <h2 style={{ fontSize: 24, margin: "0 0 16px 0" }}>Discovered Stocks</h2>

        <div style={{ display: "flex", alignItems: "center" }}>
          <RunDiscoveryButton />
          <TestTradingButton />
        </div>

        {stocks.loading && <p>Loading stocks...</p>}
        {stocks.error && <p style={{ color: "#dc2626" }}>Error: {stocks.error}</p>}

        {stocks.data.length === 0 && !stocks.loading && !stocks.error && (
          <p style={{ color: "#666" }}>
            No stocks discovered yet. Click "Run Discovery Scan" above to find explosive opportunities.
          </p>
        )}

        {stocks.data.length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ background: "#f8fafc" }}>
                  <th style={{ padding: "8px 12px", textAlign: "left", border: "1px solid #e2e8f0" }}>Symbol</th>
                  <th style={{ padding: "8px 12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Price</th>
                  <th style={{ padding: "8px 12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Score</th>
                  <th style={{ padding: "8px 12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Volume</th>
                  <th style={{ padding: "8px 12px", textAlign: "center", border: "1px solid #e2e8f0" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {stocks.data.map((stock, i) => (
                  <tr key={stock.symbol || i}>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0" }}>
                      <div>
                        <strong>{stock.symbol}</strong>
                        {stock.thesis && (
                          <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4, maxWidth: 400 }}>
                            {stock.thesis}
                          </div>
                        )}
                      </div>
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      <div>
                        <div>${(stock.price || stock.last_price || 0).toFixed(2)}</div>
                        {stock.price_target && (
                          <div style={{ fontSize: 11, color: "#10b981" }}>
                            Target: ${stock.price_target}
                          </div>
                        )}
                        {stock.stop_loss && (
                          <div style={{ fontSize: 11, color: "#ef4444" }}>
                            Stop: ${stock.stop_loss}
                          </div>
                        )}
                      </div>
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      <div>
                        <div>{(stock.composite_score || stock.score || stock.accumulation_score || stock.total_score || 0).toFixed(1)}</div>
                        {stock.risk_reward_ratio && (
                          <div style={{ fontSize: 11, color: "#3b82f6" }}>
                            R:R {stock.risk_reward_ratio}:1
                          </div>
                        )}
                      </div>
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      {stock.volume_surge ? `${stock.volume_surge}x` : (stock.volume || 0).toLocaleString()}
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "center" }}>
                      <BuyButton
                        symbol={stock.symbol}
                        price={stock.price || stock.last_price || 0}
                        stopLoss={stock.stop_loss}
                        priceTarget={stock.price_target}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);