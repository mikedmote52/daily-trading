import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { useExplosiveStocks } from "./stock-service.js";

const ordersBase = "https://alphastack-orders.onrender.com";

function BuyButton({ symbol, price }) {
  const [buying, setBuying] = useState(false);

  const handleBuy = async () => {
    if (!ordersBase) {
      alert("❌ Orders API not configured");
      return;
    }

    setBuying(true);
    try {
      const response = await fetch(`${ordersBase}/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: symbol,
          notional_usd: 1000,
          last_price: price,
          side: 'buy'
        })
      });

      if (response.ok) {
        alert(`✅ Buy order placed for ${symbol}!`);
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

  return (
    <button
      onClick={handleBuy}
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
      {buying ? "Buying..." : "BUY"}
    </button>
  );
}

function App() {
  const stocks = useExplosiveStocks();

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

        {/* Discovery Status */}
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
                        <BuyButton symbol={stock.symbol} price={stock.price} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

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