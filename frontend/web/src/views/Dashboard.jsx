// Dashboard.jsx — Live analytics dashboard with 5-second auto-refresh
import React, { useEffect, useState, useCallback } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts';

const BASE = 'http://127.0.0.1:8000';
const POLL_MS = 5000; // auto-refresh every 5 seconds

// Custom tooltip for the hourly chart
const HourTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={tooltipStyle}>
      <p style={{ fontWeight: 700, marginBottom: 4 }}>Hour {label}:00</p>
      <p style={{ color: '#6366f1' }}>Total: {payload[0]?.value ?? 0}</p>
      <p style={{ color: '#ef4444' }}>Fraud: {payload[1]?.value ?? 0}</p>
    </div>
  );
};

const Dashboard = () => {
  const [summary, setSummary] = useState({ total_transactions: 0, fraud_count: 0, safe_count: 0, fraud_rate: 0 });
  const [hourData, setHourData] = useState([]);
  const [trendData, setTrendData] = useState([]);
  const [distData, setDistData] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isLive, setIsLive] = useState(true);

  const fetchAll = useCallback(async () => {
    const get = (path) =>
      fetch(BASE + path)
        .then(r => r.json())
        .then(j => j.data)
        .catch(() => null);

    const [s, h, t, d] = await Promise.all([
      get('/api/stats/summary'),
      get('/api/stats/by_hour'),
      get('/api/stats/history_trend'),
      get('/api/stats/amount_distribution'),
    ]);

    if (s) setSummary(s);
    if (h) setHourData(h);
    if (t) setTrendData(t);
    if (d) setDistData(d);
    setLastUpdated(new Date());
  }, []);

  // Initial fetch + polling
  useEffect(() => {
    fetchAll();
    if (!isLive) return;
    const id = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(id);
  }, [fetchAll, isLive]);

  const fraudRate = summary.fraud_rate ?? 0;
  const rateColor = fraudRate > 20 ? '#ef4444' : fraudRate > 10 ? '#f59e0b' : '#22c55e';

  return (
    <div style={{ padding: '4px 0' }}>

      {/* ── Live indicator bar (hidden) ── */}
      {/* <div style={liveBarStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ ...dotStyle, background: isLive ? '#22c55e' : '#9ca3af' }} />
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#374151' }}>
            {isLive ? 'LIVE' : 'PAUSED'}
          </span>
          {lastUpdated && (
            <span style={{ fontSize: '0.78rem', color: '#9ca3af' }}>
              Last updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button id="btn-toggle-live" onClick={() => setIsLive(v => !v)} style={controlBtnStyle}>
            {isLive ? '⏸ Pause' : '▶ Resume'}
          </button>
          <button id="btn-refresh-now" onClick={fetchAll} style={{ ...controlBtnStyle, background: '#eff6ff', color: '#2563eb', borderColor: '#bfdbfe' }}>
            ↻ Refresh
          </button>
        </div>
      </div> */}

      {/* ── Metric Cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
        <MetricCard label="Transactions Analyzed" value={summary.total_transactions.toLocaleString()} color="#2563eb"  />
        <MetricCard label="Safe Transactions" value={summary.safe_count.toLocaleString()} color="#22c55e"  />
        <MetricCard label="Fraud Detected" value={summary.fraud_count.toLocaleString()} color="#ef4444" />
        <MetricCard label="Fraud Rate" value={`${summary.fraud_rate}%`} color={rateColor} />
      </div>

      {/* ── Charts Grid ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* Transactions by Hour (total + fraud overlay) */}
        <div className="visualization-card">
          <h2 className="viz-title">Transactions by Hour</h2>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={hourData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} interval={3} tickFormatter={h => `${h}:00`} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip content={<HourTooltip />} />
              <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
              <Bar dataKey="total" name="Total" fill="#6366f1" radius={[3, 3, 0, 0]} maxBarSize={20} />
              <Bar dataKey="fraud" name="Fraud" fill="#ef4444" radius={[3, 3, 0, 0]} maxBarSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Daily Transaction Trend */}
        <div className="visualization-card">
          <h2 className="viz-title">Daily Transaction Trend</h2>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={trendData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line
                type="monotone" dataKey="count" name="Transactions"
                stroke="#22c55e" strokeWidth={2.5}
                dot={{ r: 5, fill: '#22c55e' }}
                activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
          {trendData.length === 0 && <EmptyNote />}
        </div>

        {/* Amount Distribution — full width */}
        <div className="visualization-card" style={{ gridColumn: 'span 2' }}>
          <h2 className="viz-title">Transaction Amount Distribution (VND)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={distData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
              <XAxis dataKey="range" xAxisId={0} tick={{ fontSize: 12 }} />
              <XAxis dataKey="range" xAxisId={1} hide />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
              <Bar dataKey="clean" xAxisId={0} name="Clean" fill="#93c5fd" radius={[4, 4, 0, 0]} maxBarSize={80} />
              <Bar dataKey="fraud" xAxisId={1} name="Fraud" fill="rgba(239, 68, 68, 0.8)" radius={[4, 4, 0, 0]} maxBarSize={80} />
            </BarChart>
          </ResponsiveContainer>
          {distData.length > 0 && distData.every(d => d.clean === 0 && d.fraud === 0) && <EmptyNote />}
        </div>
      </div>
    </div>
  );
};

// ── Sub-components ─────────────────────────────────────────────────────────

const MetricCard = ({ label, value, color, icon }) => (
  <div className="metric-card" style={{ borderTop: `3px solid ${color}` }}>
    <span style={{ fontSize: '1.4rem' }}>{icon}</span>
    <span className="metric-label" style={{ marginTop: 8 }}>{label}</span>
    <span className="metric-value" style={{ color }}>{value}</span>
  </div>
);

const EmptyNote = () => (
  <p style={{ color: '#9ca3af', fontSize: '0.82rem', textAlign: 'center', marginTop: 8 }}>
    No data yet — run a prediction to see results.
  </p>
);

// ── Styles ─────────────────────────────────────────────────────────────────

const liveBarStyle = {
  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  background: '#fff', padding: '10px 16px', borderRadius: 10,
  border: '1px solid #e5e7eb', marginBottom: 16,
  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
};
const dotStyle = {
  width: 8, height: 8, borderRadius: '50%',
  display: 'inline-block',
  boxShadow: '0 0 0 3px rgba(34,197,94,0.25)',
  animation: 'pulse 2s infinite',
};
const controlBtnStyle = {
  padding: '5px 12px', fontSize: '0.82rem', borderRadius: 6,
  border: '1px solid #d1d5db', background: '#f9fafb',
  color: '#374151', cursor: 'pointer', fontWeight: 600,
};
const tooltipStyle = {
  background: '#fff', borderRadius: 8,
  border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
  padding: '8px 12px', fontSize: '0.85rem',
};

export default Dashboard;