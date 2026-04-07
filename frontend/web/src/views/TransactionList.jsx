// TransactionList.jsx — Transaction history with search, filter, CSV export
import React, { useState, useEffect, useMemo } from 'react';
import { useToast } from '../components/Toast';

const API_URL = 'http://127.0.0.1:8000/api/history';

const TransactionList = () => {
  const toast = useToast();
  const [transactions, setTransactions] = useState([]);
  const [search, setSearch]   = useState('');
  const [filter, setFilter]   = useState('all'); // all | fraud | safe
  const [isLoading, setIsLoading] = useState(false);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const res  = await fetch(API_URL);
      const json = await res.json();
      if (json.status === 'success') {
        setTransactions(json.data);
        toast('History refreshed.', 'success');
      }
    } catch {
      toast('Cannot reach backend — is uvicorn running?', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = async () => {
    if (!window.confirm('Clear all session transaction history?')) return;
    await fetch(API_URL, { method: 'DELETE' });
    setTransactions([]);
    toast('History cleared.', 'warning');
  };

  const exportCSV = () => {
    if (filtered.length === 0) {
      toast('No data to export.', 'warning');
      return;
    }
    const headers = ['Transaction ID', 'Timestamp', 'Amount (VND)', 'Device', 'IF Vote', 'RFR Vote', 'RFC Vote', 'Status'];
    const rows = filtered.map(tx => [
      tx.transaction_id,
      tx.timestamp,
      tx.amount,
      tx.device_use,
      tx.votes?.isolation_forest        ?? '',
      tx.votes?.random_forest_regressor ?? '',
      tx.votes?.random_forest_classifier?? '',
      tx.is_fraud ? 'Fraud' : 'Safe',
    ]);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), { href: url, download: 'transactions.csv' });
    a.click();
    URL.revokeObjectURL(url);
    toast(`Exported ${filtered.length} rows as CSV.`, 'success');
  };

  useEffect(() => { fetchHistory(); }, []);

  const filtered = useMemo(() => {
    const kw = search.toLowerCase().trim();
    return transactions.filter(tx => {
      const matchSearch =
        !kw ||
        (tx.transaction_id || '').toLowerCase().includes(kw) ||
        (tx.device_use     || '').toLowerCase().includes(kw);
      const matchFilter =
        filter === 'all' ||
        (filter === 'fraud' && tx.is_fraud) ||
        (filter === 'safe'  && !tx.is_fraud);
      return matchSearch && matchFilter;
    });
  }, [search, filter, transactions]);

  const formatAmount = v => Number(v).toLocaleString('vi-VN') + ' VND';

  // Count stats for the filter buttons
  const total = transactions.length;
  const fraudCount = transactions.filter(t => t.is_fraud).length;
  const safeCount  = total - fraudCount;

  return (
    <div className="view-transactions">

      {/* ── Controls bar ── */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '14px', flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          id="tx-search" type="text"
          className="search-input"
          placeholder="Search by Transaction ID or Device…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ flex: 1, minWidth: 200 }}
        />

        {/* Status filter pills */}
        {[
          { key: 'all',   label: `All (${total})` },
          { key: 'fraud', label: `⚠ Fraud (${fraudCount})` },
          { key: 'safe',  label: `✓ Safe (${safeCount})` },
        ].map(f => (
          <button
            key={f.key}
            id={`btn-filter-${f.key}`}
            onClick={() => setFilter(f.key)}
            style={{
              ...filterPillStyle,
              ...(filter === f.key ? activePillStyle : {}),
            }}
          >
            {f.label}
          </button>
        ))}

        <button id="btn-refresh"     onClick={fetchHistory} style={actionBtnStyle}>
          {isLoading ? '…' : '↻ Refresh'}
        </button>
        <button id="btn-export-csv"  onClick={exportCSV}   style={{ ...actionBtnStyle, background: '#eff6ff', color: '#2563eb', borderColor: '#bfdbfe' }}>
          ⬇ Export CSV
        </button>
        <button id="btn-clear-history" onClick={clearHistory} style={{ ...actionBtnStyle, background: '#fef2f2', color: '#b91c1c', borderColor: '#fecaca' }}>
          🗑 Clear
        </button>
      </div>

      {/* ── Table ── */}
      <div className="content-panel">
        <h2 className="viz-title">Transaction History ({filtered.length})</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Transaction ID</th>
                <th>Timestamp</th>
                <th>Amount</th>
                <th>Device</th>
                <th>Votes (IF / RFR / RFC)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((tx, i) => (
                <tr key={tx.transaction_id || i}>
                  <td>{tx.transaction_id}</td>
                  <td>{tx.timestamp}</td>
                  <td>{formatAmount(tx.amount)}</td>
                  <td>{tx.device_use}</td>
                  <td style={{ fontFamily: 'monospace' }}>
                    {tx.votes
                      ? `${tx.votes.isolation_forest} / ${tx.votes.random_forest_regressor} / ${tx.votes.random_forest_classifier}`
                      : '—'}
                  </td>
                  <td>
                    <span className={`badge ${tx.is_fraud ? 'high' : 'low'}`}>
                      {tx.is_fraud ? '⚠ Fraud' : '✓ Safe'}
                    </span>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', padding: '3rem', color: '#9ca3af' }}>
                    {isLoading ? 'Loading…' : 'No results. Run a prediction or change the filter.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────
const actionBtnStyle = {
  padding: '8px 14px', borderRadius: '8px', border: '1px solid #d1d5db',
  background: '#f9fafb', color: '#374151', cursor: 'pointer',
  fontWeight: 600, whiteSpace: 'nowrap',
};
const filterPillStyle = {
  padding: '6px 14px', borderRadius: '20px', border: '1px solid #e5e7eb',
  background: '#f3f4f6', color: '#6b7280', cursor: 'pointer',
  fontWeight: 600, fontSize: '0.82rem', whiteSpace: 'nowrap',
  transition: 'all 0.15s',
};
const activePillStyle = {
  background: '#1d4ed8', color: '#fff', borderColor: '#1d4ed8',
};

export default TransactionList;
