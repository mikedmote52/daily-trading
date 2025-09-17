import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

const discoveryBase = import.meta.env.VITE_DISCOVERY_API_URL;
const ordersBase    = import.meta.env.VITE_ORDERS_API_URL;

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

function BuyButton({ symbol, price }) {
  const [buying, setBuying] = useState(false);

  const handleBuy = async () => {
    if (!ordersBase) {
      alert("Orders API not configured");
      return;
    }

    setBuying(true);
    try {
      const response = await fetch(`${ordersBase.replace(/\/$/, "")}/orders/buy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          quantity: 10,
          price
        })
      });

      if (response.ok) {
        alert(`Buy order submitted for ${symbol}`);
      } else {
        alert(`Failed to submit order: ${response.status}`);
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setBuying(false);
    }
  };

  return (
    <button
      onClick={handleBuy}
      disabled={buying}
      style={{
        background: buying ? "#9ca3af" : "#16a34a",
        color: "white",
        border: "none",
        padding: "6px 12px",
        borderRadius: 4,
        cursor: buying ? "not-allowed" : "pointer",
        fontSize: 12
      }}
    >
      {buying ? "..." : "BUY"}
    </button>
  );
}

function App() {
  const d = useHealth(discoveryBase);
  const o = useHealth(ordersBase);
  const stocks = useStocks();

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui", padding: 24, lineHeight: 1.4 }}>
      <h1 style={{ fontSize: 40, margin: 0 }}>AlphaStack UI</h1>
      <p style={{ color: "#555" }}>Stock discovery and trading interface</p>

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
      </div>

      <hr style={{ margin: "24px 0" }} />

      <div>
        <h2 style={{ fontSize: 24, margin: "0 0 16px 0" }}>Discovered Stocks</h2>

        {stocks.loading && <p>Loading stocks...</p>}
        {stocks.error && <p style={{ color: "#dc2626" }}>Error: {stocks.error}</p>}

        {stocks.data.length === 0 && !stocks.loading && !stocks.error && (
          <p style={{ color: "#666" }}>
            No stocks discovered yet. Run a discovery scan or check the Discovery API.
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
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", fontWeight: "bold" }}>
                      {stock.symbol}
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      ${(stock.price || stock.last_price || 0).toFixed(2)}
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      {(stock.score || stock.total_score || 0).toFixed(2)}
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      {(stock.volume || 0).toLocaleString()}
                    </td>
                    <td style={{ padding: "8px 12px", border: "1px solid #e2e8f0", textAlign: "center" }}>
                      <BuyButton
                        symbol={stock.symbol}
                        price={stock.price || stock.last_price || 0}
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