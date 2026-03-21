import React from 'react'

/**
 * Overview View
 * Displays high-level metrics and a custom SVG line chart.
 */
const Dashboard = () => {
  return (
    <div className="view-overview">
      {/* Metric Cards Section - Replace 0s with real data from your state/API */}
      <div className="metrics-row">
        <div className="metric-card">
          <span className="metric-label">Transaction Analyzed</span>
          <span className="metric-value">0</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">High risk detected</span>
          <span className="metric-value">0</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Accounts Flagged</span>
          <span className="metric-value">0</span>
        </div>
      </div>

      {/* Visualization Card with SVG Chart */}
      <div className="visualization-card">
        <h2 className="viz-title">Visualization</h2>
        <div className="chart-container">
          <svg className="line-chart" viewBox="0 0 400 200" preserveAspectRatio="none">
            <defs>
              <linearGradient id="lineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.4"/>
                <stop offset="100%" stopColor="var(--accent)" stopOpacity="0"/>
              </linearGradient>
            </defs>
            <path className="chart-area" d=""/>
            <path className="chart-line" d=""/>
            <line className="axis" x1="0" y1="0" x2="0" y2="200" stroke="currentColor" strokeWidth="1"/>
            <line className="axis" x1="0" y1="200" x2="400" y2="200" stroke="currentColor" strokeWidth="1"/>
          </svg>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
