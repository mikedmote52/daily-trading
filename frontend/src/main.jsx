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

function App() {
  const d = useHealth(discoveryBase);
  const o = useHealth(ordersBase);

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui", padding: 24, lineHeight: 1.4 }}>
      <h1 style={{ fontSize: 40, margin: 0 }}>AlphaStack UI</h1>
      <p style={{ color: "#555" }}>Static frontend is live. Backends are wired and healthy?</p>

      <div style={{ display: "grid", gap: 12, maxWidth: 720 }}>
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

      <div style={{ fontSize: 14, color: "#333" }}>
        <p>Next: render discoveries in a table and hook "Buy" to the Orders API.</p>
        <ul>
          <li>GET <code>/health</code> (both services) → green above</li>
          <li>Set env on static site if needed:
            <code style={{ background:"#f1f5f9", padding:"2px 6px", borderRadius:4, marginLeft:6 }}>
              VITE_DISCOVERY_API_URL, VITE_ORDERS_API_URL
            </code>
          </li>
        </ul>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);