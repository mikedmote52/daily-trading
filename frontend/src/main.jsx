import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import PortfolioApp from "./portfolio.jsx";

const discoveryBase = import.meta.env.VITE_DISCOVERY_API_URL;
const ordersBase    = import.meta.env.VITE_ORDERS_API_URL;
const portfolioBase = import.meta.env.VITE_PORTFOLIO_API_URL;

function useHealth(url) {
  const [state, setState] = useState({ status: "checking", code: null });
  useEffect(() => {
    if (!url) { setState({ status: "missing", code: null }); return; }
    fetch(`${url.replace(/\/$/, "")}/health`)
      .then(r => setState({ status: r.ok ? "ok" : "error", code: r.status }))
      .catch(() => setState({ status: "error", code: null }));
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
      alert("Orders API not configured");
      return;
    }

    setBuying(true);
    try {
      const idempotencyKey = `order-${symbol}-${Date.now()}`;

      const response = await fetch(`${ordersBase.replace(/\/$/, "")}/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': idempotencyKey
        },
        body: JSON.stringify({
          ticker: symbol,
          notional_usd: actualInvestment,
          last_price: price,
          side: 'buy'
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ Buy order submitted for ${shares} shares of ${symbol}\n💰 Total: $${actualInvestment.toFixed(2)}\n🛑 Stop Loss: $${stopLossPrice.toFixed(2)}\n🎯 Profit Target: $${profitTarget.toFixed(2)}`);
        setShowModal(false);
      } else {
        const error = await response.text();
        alert(`❌ Failed to submit order: ${error}`);
      }
    } catch (err) {
      alert(`❌ Error: ${err.message}`);
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

          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handleBuy}
              disabled={buying || shares === 0}
              style={{
                flex: 1, background: buying ? "#9ca3af" : "#16a34a",
                color: "white", border: "none", padding: "10px 16px",
                borderRadius: 4, cursor: buying ? "not-allowed" : "pointer",
                fontSize: 14, fontWeight: "bold"
              }}
            >
              {buying ? "Placing Order..." : `Buy ${shares} Shares`}
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

        <RunDiscoveryButton />

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
                        <div>{(stock.score || stock.accumulation_score || stock.total_score || 0).toFixed(0)}</div>
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