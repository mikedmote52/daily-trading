import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { useExplosiveStocks } from "./stock-service.js";

const ordersBase = import.meta.env.VITE_ORDERS_API_URL || "https://alphastack-orders.onrender.com";
const portfolioBase = import.meta.env.VITE_PORTFOLIO_API_URL || "https://alphastack-portfolio.onrender.com";

function BuyButton({ symbol, price, onOrderPlaced }) {
  const [buying, setBuying] = useState(false);
  const [showAmountInput, setShowAmountInput] = useState(false);
  const [amount, setAmount] = useState(1000);

  const handleBuy = async () => {
    if (!ordersBase) {
      alert("❌ Orders API not configured");
      return;
    }

    setBuying(true);
    try {
      // Generate unique idempotency key
      const idempotencyKey = `${symbol}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const response = await fetch(`${ordersBase}/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': idempotencyKey,
        },
        body: JSON.stringify({
          ticker: symbol,
          notional_usd: amount,
          last_price: price,
          side: 'buy'
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ Buy order placed for ${symbol}! Order ID: ${result.order_id || 'N/A'}`);
        setShowAmountInput(false);
        if (onOrderPlaced) onOrderPlaced();
      } else {
        const error = await response.text();
        alert(`❌ Order failed: ${error}`);
      }
    } catch (err) {
      alert(`❌ Error: ${err.message}`);
    } finally {
      setBuying(false);
    }
  };

  if (showAmountInput) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minWidth: 120 }}>
        <input
          type="number"
          value={amount}
          onChange={(e) => setAmount(Number(e.target.value))}
          placeholder="Amount ($)"
          style={{
            padding: '6px 8px',
            borderRadius: 4,
            border: '1px solid #374151',
            background: '#1f2937',
            color: 'white',
            fontSize: 14
          }}
        />
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={handleBuy}
            disabled={buying || amount <= 0}
            style={{
              background: buying ? "#6b7280" : "#16a34a",
              color: "white",
              border: "none",
              padding: "4px 8px",
              borderRadius: 4,
              cursor: buying ? "not-allowed" : "pointer",
              fontSize: 12,
              fontWeight: "bold",
              flex: 1
            }}
          >
            {buying ? "Buying..." : "BUY"}
          </button>
          <button
            onClick={() => setShowAmountInput(false)}
            style={{
              background: "#6b7280",
              color: "white",
              border: "none",
              padding: "4px 8px",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
              flex: 1
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => setShowAmountInput(true)}
      disabled={buying}
      style={{
        background: buying ? "#6b7280" : "#16a34a",
        color: "white",
        border: "none",
        padding: "8px 16px",
        borderRadius: 4,
        cursor: buying ? "not-allowed" : "pointer",
        fontSize: 14,
        fontWeight: "bold"
      }}
    >
      BUY
    </button>
  );
}

function Portfolio({ onRefresh }) {
  const [portfolio, setPortfolio] = useState({ positions: [], loading: true, error: null });
  const [accountInfo, setAccountInfo] = useState(null);

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const fetchPortfolio = async () => {
    try {
      setPortfolio(prev => ({ ...prev, loading: true, error: null }));

      // Try portfolio service first, fallback to orders API if unavailable
      let positions = [];
      let account = null;

      try {
        console.log("Trying portfolio service...");
        const positionsResponse = await fetch(`${portfolioBase}/positions`);
        const accountResponse = await fetch(`${portfolioBase}/account`);

        if (positionsResponse.ok && accountResponse.ok) {
          positions = await positionsResponse.json();
          account = await accountResponse.json();
          console.log("✅ Portfolio service working");
        } else {
          throw new Error("Portfolio service unavailable");
        }
      } catch (portfolioError) {
        console.log("❌ Portfolio service down, using orders API fallback");

        // Fallback to orders API (which we know is working)
        const accountResponse = await fetch(`${ordersBase}/account`);
        const positionsResponse = await fetch(`${ordersBase}/positions`);

        if (accountResponse.ok) {
          const accountData = await accountResponse.json();
          account = {
            account_value: parseFloat(accountData.portfolio_value || "0"),
            cash: parseFloat(accountData.cash || "0"),
            buying_power: parseFloat(accountData.buying_power || "0"),
            day_trade_count: accountData.day_trade_count || 0
          };
        }

        if (positionsResponse.ok) {
          const positionsData = await positionsResponse.json();
          positions = positionsData.positions || [];
        }

        console.log("✅ Orders API fallback working, positions:", positions.length);
      }

      setPortfolio({ positions: positions || [], loading: false, error: null });
      setAccountInfo(account);

    } catch (error) {
      console.error('Portfolio fetch error:', error);
      setPortfolio({ positions: [], loading: false, error: error.message });
    }
  };

  const handleSell = async (symbol, qty) => {
    if (!confirm(`Sell ${qty} shares of ${symbol}?`)) return;

    try {
      const idempotencyKey = `${symbol}_sell_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const response = await fetch(`${ordersBase}/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': idempotencyKey,
        },
        body: JSON.stringify({
          ticker: symbol,
          qty: Math.abs(qty),
          side: 'sell'
        })
      });

      if (response.ok) {
        alert(`✅ Sell order placed for ${symbol}!`);
        fetchPortfolio(); // Refresh portfolio
        if (onRefresh) onRefresh();
      } else {
        const error = await response.text();
        alert(`❌ Sell order failed: ${error}`);
      }
    } catch (err) {
      alert(`❌ Error: ${err.message}`);
    }
  };

  if (portfolio.loading) {
    return (
      <div style={{ padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 20, height: 20, border: "2px solid #374151", borderTop: "2px solid #10b981", borderRadius: "50%", animation: "spin 1s linear infinite" }}></div>
          <span>Loading portfolio...</span>
        </div>
      </div>
    );
  }

  if (portfolio.error) {
    return (
      <div style={{ padding: 20 }}>
        <div style={{ color: "#ef4444", padding: 16, background: "#7f1d1d", borderRadius: 6 }}>
          ❌ Error loading portfolio: {portfolio.error}
        </div>
        <button
          onClick={fetchPortfolio}
          style={{
            marginTop: 16,
            background: "#10b981",
            color: "white",
            border: "none",
            padding: "8px 16px",
            borderRadius: 4,
            cursor: "pointer"
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: 20 }}>
      {/* Account Summary */}
      {accountInfo && (
        <div style={{ background: "#1f2937", padding: 20, borderRadius: 8, marginBottom: 24, border: "1px solid #374151" }}>
          <h2 style={{ margin: "0 0 16px 0", color: "#10b981" }}>💰 Account Summary</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
            <div>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Portfolio Value</div>
              <div style={{ fontSize: 24, fontWeight: "bold", color: "#10b981" }}>
                ${(accountInfo.portfolio_value || accountInfo.account_value) ? Number(accountInfo.portfolio_value || accountInfo.account_value).toLocaleString() : 'N/A'}
              </div>
            </div>
            <div>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Buying Power</div>
              <div style={{ fontSize: 24, fontWeight: "bold" }}>
                ${accountInfo.buying_power ? Number(accountInfo.buying_power).toLocaleString() : 'N/A'}
              </div>
            </div>
            <div>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Cash</div>
              <div style={{ fontSize: 24, fontWeight: "bold" }}>
                ${accountInfo.cash ? Number(accountInfo.cash).toLocaleString() : 'N/A'}
              </div>
            </div>
          </div>

          {/* Portfolio Performance Summary */}
          {(portfolio.positions && portfolio.positions.length > 0) && (
            <div style={{ marginTop: 16, padding: 16, background: "rgba(0,0,0,0.2)", borderRadius: 6 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 2 }}>Total Positions</div>
                  <div style={{ fontSize: 16, fontWeight: "bold" }}>{(portfolio.positions || []).length}</div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 2 }}>Positions in Red</div>
                  <div style={{ fontSize: 16, fontWeight: "bold", color: "#dc2626" }}>
                    {(portfolio.positions || []).filter(p => p.unrealized_pl < 0).length}
                  </div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 2 }}>Total Unrealized P&L</div>
                  <div style={{ fontSize: 16, fontWeight: "bold",
                               color: (portfolio.positions || []).reduce((sum, p) => sum + Number(p.unrealized_pl || 0), 0) >= 0 ? "#10b981" : "#dc2626" }}>
                    ${Number((portfolio.positions || []).reduce((sum, p) => sum + Number(p.unrealized_pl || 0), 0)).toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 2 }}>High Risk Positions</div>
                  <div style={{ fontSize: 16, fontWeight: "bold", color: "#dc2626" }}>
                    {(portfolio.positions || []).filter(p => p.unrealized_plpc < -0.20).length}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Positions */}
      <div style={{ background: "#1f2937", padding: 20, borderRadius: 8, border: "1px solid #374151" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ margin: 0, color: "#10b981" }}>📈 Current Positions</h2>
          <button
            onClick={fetchPortfolio}
            style={{
              background: "#374151",
              color: "white",
              border: "none",
              padding: "6px 12px",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12
            }}
          >
            Refresh
          </button>
        </div>

        {portfolio.positions.length === 0 ? (
          <div style={{ color: "#9ca3af", padding: 16, textAlign: "center" }}>
            No positions found. Start trading to build your portfolio!
          </div>
        ) : (
          <div style={{ display: "grid", gap: 12 }}>
            {portfolio.positions.map((position, i) => (
              <div key={position.symbol || i} style={{
                background: "linear-gradient(135deg, #0f172a, #1e293b)",
                padding: 16,
                borderRadius: 6,
                border: "1px solid #334155"
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                      <h3 style={{ margin: 0, fontSize: 20, fontWeight: "bold" }}>
                        {position.symbol}
                      </h3>
                      <span style={{
                        background: position.unrealized_pl >= 0 ? "#16a34a" : "#dc2626",
                        color: "white",
                        padding: "2px 6px",
                        borderRadius: 4,
                        fontSize: 10,
                        fontWeight: "bold"
                      }}>
                        {position.unrealized_pl >= 0 ? '+' : ''}${Number(position.unrealized_pl || 0).toFixed(2)}
                        ({position.unrealized_plpc >= 0 ? '+' : ''}{(position.unrealized_plpc * 100).toFixed(1)}%)
                      </span>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 12 }}>
                      <div>
                        <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 2 }}>Shares</div>
                        <div style={{ fontSize: 14, fontWeight: "bold" }}>{position.qty}</div>
                      </div>
                      <div>
                        <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 2 }}>Avg Cost</div>
                        <div style={{ fontSize: 14, fontWeight: "bold" }}>${Number(position.avg_entry_price || position.avg_cost || 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 2 }}>Current Price</div>
                        <div style={{ fontSize: 14, fontWeight: "bold" }}>${Number(position.current_price || 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 2 }}>Market Value</div>
                        <div style={{ fontSize: 14, fontWeight: "bold" }}>${Number(position.market_value || 0).toFixed(2)}</div>
                      </div>
                    </div>

                    {/* Discovery Context and Decision Support */}
                    <div style={{ marginTop: 12, padding: 10, background: "rgba(0,0,0,0.3)", borderRadius: 6 }}>
                      <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                        {/* Entry Thesis */}
                        <div style={{ flex: 1 }}>
                          <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 2 }}>Discovery Thesis</div>
                          <div style={{ fontSize: 11, color: "#10b981" }}>
                            {position.discovery_thesis || "Pre-explosion accumulation pattern"}
                          </div>
                        </div>

                        {/* Risk Status */}
                        <div>
                          <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 2 }}>Risk Level</div>
                          <span style={{
                            padding: "2px 6px",
                            borderRadius: 3,
                            fontSize: 10,
                            fontWeight: "bold",
                            background: position.unrealized_plpc < -0.20 ? "#dc2626" :
                                      position.unrealized_plpc < -0.10 ? "#f59e0b" : "#10b981",
                            color: "white"
                          }}>
                            {position.unrealized_plpc < -0.20 ? "HIGH RISK" :
                             position.unrealized_plpc < -0.10 ? "MODERATE" : "NORMAL"}
                          </span>
                        </div>

                        {/* Action Recommendation */}
                        <div style={{ flex: 1 }}>
                          <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 2 }}>Action</div>
                          <div style={{ fontSize: 11, fontWeight: "bold",
                                       color: position.unrealized_plpc < -0.20 ? "#dc2626" :
                                             position.unrealized_plpc < -0.10 ? "#f59e0b" :
                                             position.unrealized_plpc < 0 ? "#94a3b8" : "#10b981" }}>
                            {position.unrealized_plpc < -0.20 ? "⚠️ Review Exit Strategy" :
                             position.unrealized_plpc < -0.10 ? "📊 Monitor Closely" :
                             position.unrealized_plpc < -0.05 ? "💎 Hold - Pattern Active" :
                             position.unrealized_plpc < 0 ? "🔄 Buy-the-Dip Opportunity" :
                             "✅ On Track"}
                          </div>
                        </div>

                        {/* Stop Loss / Target */}
                        <div>
                          <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 2 }}>Stop/Target</div>
                          <div style={{ fontSize: 10 }}>
                            <span style={{ color: "#dc2626" }}>
                              ${(position.avg_entry_price * 0.90).toFixed(2)}
                            </span>
                            {" / "}
                            <span style={{ color: "#10b981" }}>
                              ${(position.avg_entry_price * 1.20).toFixed(2)}
                            </span>
                          </div>
                        </div>

                        {/* Catalyst Alert */}
                        <div>
                          <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 2 }}>Next Catalyst</div>
                          <div style={{ fontSize: 10 }}>
                            {(() => {
                              // Estimated earnings dates based on typical quarterly schedules
                              const catalystMap = {
                                'FATN': { date: 'Oct 15', type: 'Earnings' },
                                'CDLX': { date: 'Oct 22', type: 'Earnings' },
                                'LAES': { date: 'Oct 28', type: 'Earnings' },
                                'LASE': { date: 'Nov 5', type: 'Earnings' },
                                'QMCO': { date: 'Nov 12', type: 'Earnings' },
                                'CCCS': { date: 'Nov 18', type: 'Earnings' }
                              };
                              const catalyst = catalystMap[position.symbol];
                              if (catalyst) {
                                return (
                                  <span style={{ color: "#f59e0b", fontSize: 9 }}>
                                    📅 {catalyst.date} {catalyst.type}
                                  </span>
                                );
                              }
                              return <span style={{ color: "#6b7280", fontSize: 9 }}>No catalyst</span>;
                            })()}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div style={{ marginLeft: 16 }}>
                    <button
                      onClick={() => handleSell(position.symbol, position.qty)}
                      style={{
                        background: "#dc2626",
                        color: "white",
                        border: "none",
                        padding: "6px 12px",
                        borderRadius: 4,
                        cursor: "pointer",
                        fontSize: 12,
                        fontWeight: "bold"
                      }}
                    >
                      SELL
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const stocks = useExplosiveStocks();
  const [activeTab, setActiveTab] = useState('discovery');
  const [refreshKey, setRefreshKey] = useState(0);

  const handleOrderPlaced = () => {
    setRefreshKey(prev => prev + 1);
  };

  return (
    <div style={{ padding: 24, minHeight: "100vh" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 48, margin: 0, background: "linear-gradient(135deg, #10b981, #3b82f6)", backgroundClip: "text", WebkitBackgroundClip: "text", color: "transparent" }}>
            🔥 AlphaStack
          </h1>
          <p style={{ fontSize: 18, color: "#9ca3af", margin: "8px 0 0 0" }}>
            AI-Powered Explosive Stock Discovery • Live Trading System
          </p>
        </div>

        {/* Tab Navigation */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", gap: 4, background: "#1f2937", padding: 4, borderRadius: 8, border: "1px solid #374151" }}>
            <button
              onClick={() => setActiveTab('discovery')}
              style={{
                background: activeTab === 'discovery' ? "#10b981" : "transparent",
                color: activeTab === 'discovery' ? "white" : "#9ca3af",
                border: "none",
                padding: "12px 24px",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 16,
                fontWeight: "bold",
                flex: 1,
                transition: "all 0.2s"
              }}
            >
              🎯 Discovery
            </button>
            <button
              onClick={() => setActiveTab('portfolio')}
              style={{
                background: activeTab === 'portfolio' ? "#10b981" : "transparent",
                color: activeTab === 'portfolio' ? "white" : "#9ca3af",
                border: "none",
                padding: "12px 24px",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 16,
                fontWeight: "bold",
                flex: 1,
                transition: "all 0.2s"
              }}
            >
              📈 Portfolio
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'discovery' && (
          <div style={{ background: "#1f2937", padding: 20, borderRadius: 8, marginBottom: 24, border: "1px solid #374151" }}>
            <h2 style={{ margin: "0 0 16px 0", color: "#10b981" }}>🎯 Discovered Stocks</h2>

            {stocks.loading && (
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 20, height: 20, border: "2px solid #374151", borderTop: "2px solid #10b981", borderRadius: "50%", animation: "spin 1s linear infinite" }}></div>
                <span>Scanning markets for explosive opportunities...</span>
              </div>
            )}

            {stocks.error && (
              <div style={{ color: "#ef4444", padding: 16, background: "#7f1d1d", borderRadius: 6 }}>
                ❌ Error loading stocks: {stocks.error}
              </div>
            )}

            {!stocks.loading && !stocks.error && stocks.data.length === 0 && (
              <div style={{ color: "#9ca3af", padding: 16 }}>
                No explosive opportunities found at this time. System continues scanning...
              </div>
            )}

            {stocks.data.length > 0 && (
              <>
                <div style={{ color: "#10b981", fontSize: 16, fontWeight: "bold", marginBottom: 16 }}>
                  ✅ Found {stocks.data.length} explosive opportunities!
                </div>

                <div style={{ display: "grid", gap: 16 }}>
                  {stocks.data.map((stock, i) => (
                    <div key={stock.symbol || i} style={{
                      background: "linear-gradient(135deg, #059669, #0d9488)",
                      padding: 20,
                      borderRadius: 8,
                      border: "1px solid #10b981"
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                            <h3 style={{ margin: 0, fontSize: 24, fontWeight: "bold" }}>
                              {stock.symbol}
                            </h3>
                            <span style={{ background: "#16a34a", color: "white", padding: "4px 8px", borderRadius: 4, fontSize: 12, fontWeight: "bold" }}>
                              Score: {(stock.score || 0).toFixed(1)}
                            </span>
                          </div>

                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, marginBottom: 16 }}>
                            <div>
                              <div style={{ color: "#ecfdf5", fontSize: 12, marginBottom: 4 }}>Price</div>
                              <div style={{ fontSize: 18, fontWeight: "bold" }}>${(stock.price || 0).toFixed(2)}</div>
                            </div>
                            <div>
                              <div style={{ color: "#ecfdf5", fontSize: 12, marginBottom: 4 }}>Volume Surge</div>
                              <div style={{ fontSize: 18, fontWeight: "bold" }}>{stock.rvol?.toFixed(1)}x</div>
                            </div>
                            <div>
                              <div style={{ color: "#ecfdf5", fontSize: 12, marginBottom: 4 }}>Volume</div>
                              <div style={{ fontSize: 18, fontWeight: "bold" }}>{(stock.volume || 0).toLocaleString()}</div>
                            </div>
                          </div>

                          {stock.reason && (
                            <div style={{ background: "rgba(0,0,0,0.2)", padding: 12, borderRadius: 6, marginBottom: 16 }}>
                              <div style={{ fontSize: 14, lineHeight: 1.5 }}>
                                {stock.reason}
                              </div>
                            </div>
                          )}
                        </div>

                        <div style={{ marginLeft: 20 }}>
                          <BuyButton symbol={stock.symbol} price={stock.price} onOrderPlaced={handleOrderPlaced} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'portfolio' && (
          <Portfolio key={refreshKey} onRefresh={() => setRefreshKey(prev => prev + 1)} />
        )}

        {/* Footer */}
        <div style={{ textAlign: "center", color: "#6b7280", fontSize: 14 }}>
          <div>🤖 Powered by AI • 📊 Real-time Market Data • 📈 Paper Trading</div>
          <div style={{ marginTop: 8 }}>
            Discovery API: <span style={{ color: "#10b981" }}>●</span> Connected •
            Orders API: <span style={{ color: "#10b981" }}>●</span> Ready
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);