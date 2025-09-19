import React, { useEffect, useState } from "react";

const portfolioBase = import.meta.env.VITE_PORTFOLIO_API_URL || 'http://localhost:8002';

function usePortfolioData(endpoint, interval = 30000) {
  const [data, setData] = useState({ data: null, loading: true, error: null });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${portfolioBase}${endpoint}`);
        if (response.ok) {
          const result = await response.json();
          setData({ data: result, loading: false, error: null });
        } else {
          setData({ data: null, loading: false, error: `API Error: ${response.status}` });
        }
      } catch (err) {
        setData({ data: null, loading: false, error: err.message });
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [endpoint, interval]);

  return data;
}

function HealthBar({ value, label, color }) {
  const getColor = () => {
    if (color) return color;
    if (value >= 80) return "#10b981"; // Green
    if (value >= 60) return "#f59e0b"; // Yellow
    if (value >= 40) return "#f97316"; // Orange
    return "#ef4444"; // Red
  };

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
        <span>{label}</span>
        <span>{value.toFixed(1)}</span>
      </div>
      <div style={{ width: "100%", height: 8, backgroundColor: "#e5e7eb", borderRadius: 4 }}>
        <div
          style={{
            width: `${Math.min(value, 100)}%`,
            height: "100%",
            backgroundColor: getColor(),
            borderRadius: 4,
            transition: "width 0.3s ease"
          }}
        />
      </div>
    </div>
  );
}

function StatusBadge({ status, urgency, action }) {
  const getStatusColor = (status) => {
    switch (status) {
      case "ON_TRACK": return "#10b981";
      case "AHEAD": return "#059669";
      case "BEHIND": return "#f59e0b";
      case "FAILED": return "#ef4444";
      default: return "#6b7280";
    }
  };

  const getActionColor = (action) => {
    switch (action) {
      case "BUY": case "ADD": return "#10b981";
      case "HOLD": return "#6b7280";
      case "TRIM": return "#f59e0b";
      case "SELL": case "EXIT": return "#ef4444";
      default: return "#6b7280";
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case "CRITICAL": return "#dc2626";
      case "HIGH": return "#f97316";
      case "MEDIUM": return "#f59e0b";
      case "LOW": return "#10b981";
      default: return "#6b7280";
    }
  };

  const badgeStyle = {
    padding: "2px 8px",
    borderRadius: 12,
    fontSize: 10,
    fontWeight: "bold",
    color: "white",
    marginRight: 4
  };

  return (
    <div style={{ display: "flex", gap: 4 }}>
      {status && (
        <span style={{ ...badgeStyle, backgroundColor: getStatusColor(status) }}>
          {status}
        </span>
      )}
      {action && (
        <span style={{ ...badgeStyle, backgroundColor: getActionColor(action) }}>
          {action}
        </span>
      )}
      {urgency && (
        <span style={{ ...badgeStyle, backgroundColor: getUrgencyColor(urgency) }}>
          {urgency}
        </span>
      )}
    </div>
  );
}

function PortfolioDashboard() {
  const summary = usePortfolioData("/portfolio/summary");
  const health = usePortfolioData("/portfolio/health");
  const alerts = usePortfolioData("/portfolio/alerts", 15000); // Check alerts every 15 seconds

  if (summary.loading) {
    return <div style={{ padding: 24 }}>Loading portfolio dashboard...</div>;
  }

  if (summary.error) {
    return (
      <div style={{ padding: 24, color: "#ef4444" }}>
        Error loading portfolio: {summary.error}
      </div>
    );
  }

  const summaryData = summary.data;
  const healthData = health.data;
  const alertsData = alerts.data;

  return (
    <div style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h1 style={{ fontSize: 32, margin: "0 0 24px 0" }}>Portfolio Dashboard</h1>
      
      {/* Portfolio Summary */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", 
        gap: 16, 
        marginBottom: 32 
      }}>
        <div style={{ background: "#f8fafc", padding: 16, borderRadius: 8 }}>
          <h3 style={{ margin: "0 0 8px 0", fontSize: 14, color: "#6b7280" }}>Total Value</h3>
          <div style={{ fontSize: 24, fontWeight: "bold" }}>
            ${summaryData.total_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
        </div>
        
        <div style={{ background: "#f8fafc", padding: 16, borderRadius: 8 }}>
          <h3 style={{ margin: "0 0 8px 0", fontSize: 14, color: "#6b7280" }}>Total P&L</h3>
          <div style={{ 
            fontSize: 24, 
            fontWeight: "bold", 
            color: summaryData.total_pnl >= 0 ? "#10b981" : "#ef4444" 
          }}>
            {summaryData.total_pnl >= 0 ? "+" : ""}${summaryData.total_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            <span style={{ fontSize: 16, marginLeft: 8 }}>
              ({summaryData.total_pnl_percent >= 0 ? "+" : ""}{summaryData.total_pnl_percent.toFixed(1)}%)
            </span>
          </div>
        </div>
        
        <div style={{ background: "#f8fafc", padding: 16, borderRadius: 8 }}>
          <h3 style={{ margin: "0 0 8px 0", fontSize: 14, color: "#6b7280" }}>Positions</h3>
          <div style={{ fontSize: 24, fontWeight: "bold" }}>{summaryData.position_count}</div>
        </div>
        
        <div style={{ background: "#f8fafc", padding: 16, borderRadius: 8 }}>
          <h3 style={{ margin: "0 0 8px 0", fontSize: 14, color: "#6b7280" }}>Cash</h3>
          <div style={{ fontSize: 24, fontWeight: "bold" }}>
            ${summaryData.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
        </div>
      </div>

      {/* Alerts Section */}
      {alertsData && alertsData.total_alerts > 0 && (
        <div style={{ 
          background: "#fef2f2", 
          border: "1px solid #fecaca", 
          borderRadius: 8, 
          padding: 16, 
          marginBottom: 24 
        }}>
          <h3 style={{ margin: "0 0 12px 0", color: "#dc2626" }}>⚠️ Portfolio Alerts ({alertsData.total_alerts})</h3>
          <div style={{ display: "grid", gap: 8 }}>
            {alertsData.alerts.slice(0, 5).map((alert, i) => (
              <div key={i} style={{ 
                background: "white", 
                padding: 8, 
                borderRadius: 4, 
                fontSize: 14,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center"
              }}>
                <span>{alert.message}</span>
                <StatusBadge urgency={alert.urgency} action={alert.action} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Portfolio Health */}
      {healthData && (
        <div style={{ 
          background: "white", 
          border: "1px solid #e5e7eb", 
          borderRadius: 8, 
          padding: 16, 
          marginBottom: 24 
        }}>
          <h3 style={{ margin: "0 0 16px 0" }}>Portfolio Health</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: 16 }}>
            <div>
              <h4 style={{ margin: "0 0 12px 0", fontSize: 14, color: "#6b7280" }}>Overall Health</h4>
              <HealthBar value={healthData.overall_health} label="Overall Score" />
              <HealthBar value={healthData.technical_health} label="Technical" />
              <HealthBar value={healthData.fundamental_health} label="Fundamental" />
            </div>
            
            <div>
              <h4 style={{ margin: "0 0 12px 0", fontSize: 14, color: "#6b7280" }}>Risk Metrics</h4>
              <HealthBar 
                value={100 - healthData.concentration_risk} 
                label="Diversification" 
                color="#3b82f6"
              />
              <HealthBar 
                value={100 - healthData.volatility_score} 
                label="Stability" 
                color="#8b5cf6"
              />
            </div>
            
            <div>
              <h4 style={{ margin: "0 0 12px 0", fontSize: 14, color: "#6b7280" }}>Performance</h4>
              <div style={{ fontSize: 14, marginBottom: 8 }}>
                <strong>Win Rate:</strong> {healthData.win_rate.toFixed(1)}%
              </div>
              <div style={{ fontSize: 14, marginBottom: 8 }}>
                <strong>Avg Winner:</strong> +{healthData.avg_winner.toFixed(1)}%
              </div>
              <div style={{ fontSize: 14, marginBottom: 8 }}>
                <strong>Max Drawdown:</strong> {healthData.max_drawdown.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Top/Worst Performers */}
      {summaryData.top_performer && summaryData.worst_performer && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
          <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: 16 }}>
            <h3 style={{ margin: "0 0 8px 0", color: "#166534" }}>🏆 Top Performer</h3>
            <div style={{ fontSize: 18, fontWeight: "bold" }}>{summaryData.top_performer.symbol}</div>
            <div style={{ color: "#10b981", fontSize: 16 }}>
              +{summaryData.top_performer.pnl_percent.toFixed(1)}%
            </div>
          </div>
          
          <div style={{ background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, padding: 16 }}>
            <h3 style={{ margin: "0 0 8px 0", color: "#dc2626" }}>📉 Worst Performer</h3>
            <div style={{ fontSize: 18, fontWeight: "bold" }}>{summaryData.worst_performer.symbol}</div>
            <div style={{ color: "#ef4444", fontSize: 16 }}>
              {summaryData.worst_performer.pnl_percent.toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PositionsTable() {
  const positions = usePortfolioData("/portfolio/positions");
  const recommendations = usePortfolioData("/portfolio/recommendations");
  
  if (positions.loading) {
    return <div style={{ padding: 24 }}>Loading positions...</div>;
  }

  if (positions.error) {
    return (
      <div style={{ padding: 24, color: "#ef4444" }}>
        Error loading positions: {positions.error}
      </div>
    );
  }

  const positionsData = positions.data || [];
  const recommendationsData = recommendations.data || [];
  
  // Create recommendations lookup
  const recommendationsMap = {};
  recommendationsData.forEach(rec => {
    recommendationsMap[rec.symbol] = rec;
  });

  return (
    <div style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h2 style={{ fontSize: 28, margin: "0 0 24px 0" }}>Portfolio Positions</h2>
      
      {positionsData.length === 0 ? (
        <div style={{ padding: 40, textAlign: "center", color: "#6b7280" }}>
          No positions found. Start trading to see your portfolio here.
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ background: "#f8fafc" }}>
                <th style={{ padding: "12px", textAlign: "left", border: "1px solid #e2e8f0" }}>Symbol</th>
                <th style={{ padding: "12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Shares</th>
                <th style={{ padding: "12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Avg Price</th>
                <th style={{ padding: "12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Current</th>
                <th style={{ padding: "12px", textAlign: "right", border: "1px solid #e2e8f0" }}>Market Value</th>
                <th style={{ padding: "12px", textAlign: "right", border: "1px solid #e2e8f0" }}>P&L</th>
                <th style={{ padding: "12px", textAlign: "center", border: "1px solid #e2e8f0" }}>Health</th>
                <th style={{ padding: "12px", textAlign: "center", border: "1px solid #e2e8f0" }}>Status</th>
                <th style={{ padding: "12px", textAlign: "center", border: "1px solid #e2e8f0" }}>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {positionsData.map((position, i) => {
                const recommendation = recommendationsMap[position.symbol];
                return (
                  <tr key={position.symbol}>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0" }}>
                      <div>
                        <strong>{position.symbol}</strong>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          {position.days_held} days held • {position.weight.toFixed(1)}% weight
                        </div>
                        {position.price_target && (
                          <div style={{ fontSize: 11, color: "#10b981" }}>
                            Target: ${position.price_target}
                          </div>
                        )}
                      </div>
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      {position.shares.toLocaleString()}
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      ${position.avg_price.toFixed(2)}
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      ${position.current_price.toFixed(2)}
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "right" }}>
                      ${position.market_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </td>
                    <td style={{ 
                      padding: "12px", 
                      border: "1px solid #e2e8f0", 
                      textAlign: "right",
                      color: position.unrealized_pnl >= 0 ? "#10b981" : "#ef4444"
                    }}>
                      <div>
                        {position.unrealized_pnl >= 0 ? "+" : ""}${position.unrealized_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </div>
                      <div style={{ fontSize: 12 }}>
                        ({position.unrealized_pnl_percent >= 0 ? "+" : ""}{position.unrealized_pnl_percent.toFixed(1)}%)
                      </div>
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "center" }}>
                      <div style={{ minWidth: 80 }}>
                        <HealthBar value={position.overall_health} label="" />
                        <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                          T:{position.technical_health.toFixed(0)} F:{position.fundamental_health.toFixed(0)} Th:{position.thesis_health.toFixed(0)}
                        </div>
                      </div>
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "center" }}>
                      <StatusBadge status={position.thesis_performance} />
                      {position.max_drawdown > 10 && (
                        <div style={{ fontSize: 11, color: "#ef4444", marginTop: 4 }}>
                          Max DD: {position.max_drawdown.toFixed(1)}%
                        </div>
                      )}
                    </td>
                    <td style={{ padding: "12px", border: "1px solid #e2e8f0", textAlign: "center" }}>
                      {recommendation ? (
                        <div>
                          <StatusBadge action={recommendation.action} urgency={recommendation.urgency} />
                          <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                            {recommendation.confidence}% confidence
                          </div>
                          <div style={{ fontSize: 10, color: "#6b7280", marginTop: 2, maxWidth: 200 }}>
                            {recommendation.rationale}
                          </div>
                        </div>
                      ) : (
                        <span style={{ color: "#9ca3af" }}>Analyzing...</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function RecommendationsPanel() {
  const recommendations = usePortfolioData("/portfolio/recommendations");
  
  if (recommendations.loading) {
    return <div style={{ padding: 24 }}>Loading recommendations...</div>;
  }

  if (recommendations.error) {
    return (
      <div style={{ padding: 24, color: "#ef4444" }}>
        Error loading recommendations: {recommendations.error}
      </div>
    );
  }

  const recommendationsData = recommendations.data || [];
  
  // Group by urgency
  const critical = recommendationsData.filter(r => r.urgency === "CRITICAL");
  const high = recommendationsData.filter(r => r.urgency === "HIGH");
  const medium = recommendationsData.filter(r => r.urgency === "MEDIUM");
  const low = recommendationsData.filter(r => r.urgency === "LOW");

  return (
    <div style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h2 style={{ fontSize: 28, margin: "0 0 24px 0" }}>AI Recommendations</h2>
      
      {recommendationsData.length === 0 ? (
        <div style={{ padding: 40, textAlign: "center", color: "#6b7280" }}>
          No recommendations available. Portfolio analysis in progress.
        </div>
      ) : (
        <div>
          {/* Critical Recommendations */}
          {critical.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: "#dc2626", marginBottom: 16 }}>🚨 Critical Actions Required</h3>
              <div style={{ display: "grid", gap: 12 }}>
                {critical.map(rec => (
                  <div key={rec.symbol} style={{
                    background: "#fef2f2",
                    border: "1px solid #fecaca",
                    borderRadius: 8,
                    padding: 16
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <h4 style={{ margin: 0, fontSize: 18 }}>{rec.symbol}</h4>
                      <StatusBadge action={rec.action} urgency={rec.urgency} />
                    </div>
                    <p style={{ margin: "0 0 8px 0", fontSize: 14 }}>{rec.rationale}</p>
                    <div style={{ fontSize: 12, color: "#6b7280" }}>
                      Confidence: {rec.confidence}%
                      {rec.suggested_shares && ` • Suggested: ${rec.suggested_shares} shares`}
                      {rec.target_weight && ` • Target Weight: ${(rec.target_weight * 100).toFixed(1)}%`}
                    </div>
                    {rec.risk_factors && rec.risk_factors.length > 0 && (
                      <div style={{ marginTop: 8, fontSize: 12 }}>
                        <strong>Risk Factors:</strong> {rec.risk_factors.join(", ")}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* High Priority */}
          {high.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: "#f97316", marginBottom: 16 }}>⚡ High Priority</h3>
              <div style={{ display: "grid", gap: 12 }}>
                {high.map(rec => (
                  <div key={rec.symbol} style={{
                    background: "#fffbeb",
                    border: "1px solid #fed7aa",
                    borderRadius: 8,
                    padding: 16
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <h4 style={{ margin: 0 }}>{rec.symbol}</h4>
                      <StatusBadge action={rec.action} urgency={rec.urgency} />
                    </div>
                    <p style={{ margin: "0 0 8px 0", fontSize: 14 }}>{rec.rationale}</p>
                    <div style={{ fontSize: 12, color: "#6b7280" }}>
                      Confidence: {rec.confidence}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Medium Priority */}
          {medium.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: "#f59e0b", marginBottom: 16 }}>📊 Medium Priority</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12 }}>
                {medium.map(rec => (
                  <div key={rec.symbol} style={{
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                    borderRadius: 8,
                    padding: 12
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <h4 style={{ margin: 0, fontSize: 16 }}>{rec.symbol}</h4>
                      <StatusBadge action={rec.action} />
                    </div>
                    <p style={{ margin: "0 0 4px 0", fontSize: 13 }}>{rec.rationale}</p>
                    <div style={{ fontSize: 11, color: "#6b7280" }}>
                      {rec.confidence}% confidence
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Low Priority */}
          {low.length > 0 && (
            <div>
              <h3 style={{ color: "#10b981", marginBottom: 16 }}>✅ Low Priority</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: 8 }}>
                {low.map(rec => (
                  <div key={rec.symbol} style={{
                    background: "#f0fdf4",
                    border: "1px solid #bbf7d0",
                    borderRadius: 6,
                    padding: 10
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontWeight: "bold" }}>{rec.symbol}</span>
                      <StatusBadge action={rec.action} />
                    </div>
                    <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>
                      {rec.rationale}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PortfolioApp() {
  const [activeTab, setActiveTab] = useState("dashboard");

  const tabStyle = (isActive) => ({
    padding: "12px 24px",
    border: "none",
    background: isActive ? "#2563eb" : "#f3f4f6",
    color: isActive ? "white" : "#374151",
    cursor: "pointer",
    borderRadius: "6px 6px 0 0",
    fontSize: 14,
    fontWeight: "500"
  });

  return (
    <div style={{ minHeight: "100vh", background: "#f9fafb" }}>
      {/* Navigation Tabs */}
      <div style={{ background: "white", borderBottom: "1px solid #e5e7eb" }}>
        <div style={{ padding: "0 24px", display: "flex", gap: 4 }}>
          <button
            style={tabStyle(activeTab === "dashboard")}
            onClick={() => setActiveTab("dashboard")}
          >
            📊 Dashboard
          </button>
          <button
            style={tabStyle(activeTab === "positions")}
            onClick={() => setActiveTab("positions")}
          >
            💼 Positions
          </button>
          <button
            style={tabStyle(activeTab === "recommendations")}
            onClick={() => setActiveTab("recommendations")}
          >
            🤖 AI Recommendations
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === "dashboard" && <PortfolioDashboard />}
        {activeTab === "positions" && <PositionsTable />}
        {activeTab === "recommendations" && <RecommendationsPanel />}
      </div>
    </div>
  );
}

export default PortfolioApp;
