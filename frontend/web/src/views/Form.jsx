// Form.jsx — Predict Fraud with full inline validation and Toast feedback
import React, { useState } from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { useToast } from '../components/Toast';

ChartJS.register(ArcElement, Tooltip, Legend);

const WORKING_STATUSES = ['Employed', 'Self-employed', 'Student', 'Retired', 'Unemployed', 'Freelancer'];
const DEVICES = ['Mobile', 'Desktop', 'MacBook Air', 'Tablet', 'Android Phone', 'iPhone 15', 'Samsung S23', 'Web Browser', 'Other'];
const GENDERS = ['Female', 'Male'];
const API_URL = 'http://127.0.0.1:8000/api/predicts';

const initialCustomer = { customerId: '', dob: '', gender: 'Female', location: '', workingStatus: 'Employed', salary: '' };
const initialTransaction = {
  transactionId: '', senderId: '', receiverId: '',
  amount: '', accountBalance: '',
  date: new Date().toISOString().split('T')[0],
  hour: '12', minute: '00',
  detail: 'Payment', device: 'Mobile',
  geological: '10.762622, 106.660172',
};

// ── Validation rules ──────────────────────────────────────────────────────
const validateCustomer = (c) => {
  const errs = {};
  if (!c.customerId.trim())   errs.customerId = 'Customer ID is required.';
  if (!c.dob)                 errs.dob = 'Date of Birth is required.';
  else if (new Date(c.dob) >= new Date()) errs.dob = 'Date of Birth must be in the past.';
  if (!c.location.trim())     errs.location = 'Location is required.';
  if (!c.salary)              errs.salary = 'Salary is required.';
  else if (Number(c.salary) < 0) errs.salary = 'Salary must be a positive number.';
  return errs;
};

const validateTransaction = (t) => {
  const errs = {};
  if (!t.transactionId.trim())  errs.transactionId = 'Transaction ID is required.';
  if (!t.amount)                errs.amount = 'Amount is required.';
  else if (Number(t.amount) <= 0) errs.amount = 'Amount must be greater than 0.';
  if (!t.accountBalance)         errs.accountBalance = 'Account balance is required.';
  else if (Number(t.accountBalance) < 0) errs.accountBalance = 'Balance must be non-negative.';
  const h = Number(t.hour);
  if (!t.hour || h < 0 || h > 23) errs.hour = 'Hour must be 0–23.';
  const m = Number(t.minute);
  if (!t.minute || m < 0 || m > 59) errs.minute = 'Minute must be 0–59.';
  if (!t.detail.trim())         errs.detail = 'Transaction detail is required.';
  return errs;
};

const Form = () => {
  const toast = useToast();
  const [customer, setCustomerState]         = useState(initialCustomer);
  const [transaction, setTransactionState]   = useState(initialTransaction);
  const [cErrors, setCErrors] = useState({});
  const [tErrors, setTErrors] = useState({});
  const [result, setResult]   = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const setCustomer    = patch => setCustomerState(c => ({ ...c, ...patch }));
  const setTransaction = patch => setTransactionState(t => ({ ...t, ...patch }));

  // Live validation: clear individual error as user corrects a field
  const onCustomerChange = (key, value) => {
    setCustomer({ [key]: value });
    if (cErrors[key]) setCErrors(e => { const n = { ...e }; delete n[key]; return n; });
  };
  const onTransactionChange = (key, value) => {
    setTransaction({ [key]: value });
    if (tErrors[key]) setTErrors(e => { const n = { ...e }; delete n[key]; return n; });
  };

  const buildPayload = () => {
    const timestamp = `${transaction.date} ${transaction.hour.padStart(2, '0')}:${transaction.minute.padStart(2, '0')}:00`;
    return {
      customer: {
        'Customer ID': customer.customerId,
        'Date of Birth': customer.dob,
        'Gender': customer.gender,
        'Location': customer.location,
        'Working Status': customer.workingStatus,
        'Salary (per month)': Number(customer.salary) || 0,
      },
      transaction: {
        'Transaction ID':      transaction.transactionId,
        'Timestamp':           timestamp,
        'Sender Account ID':   transaction.senderId || 'N/A',
        'Receiver Account ID': transaction.receiverId || 'N/A',
        'Transaction amount':  Number(transaction.amount) || 0,
        'Transaction Detail':  transaction.detail || 'Payment',
        'Geological':          transaction.geological,
        'Device Use':          transaction.device,
        'Account balance':     Number(transaction.accountBalance) || 0,
      },
    };
  };

  const handlePredict = async () => {
    // Validate both sections
    const ce = validateCustomer(customer);
    const te = validateTransaction(transaction);
    setCErrors(ce);
    setTErrors(te);
    if (Object.keys(ce).length || Object.keys(te).length) {
      toast('Please fix the highlighted fields before submitting.', 'warning');
      return;
    }

    setIsLoading(true);
    try {
      const res  = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      });
      const json = await res.json();
      if (json.status === 'success') {
        setResult(json.data);
        toast(
          json.data.is_fraud ? '⚠ HIGH RISK transaction detected!' : '✅ Transaction appears safe.',
          json.data.is_fraud ? 'error' : 'success'
        );
      } else {
        toast('Backend error: ' + JSON.stringify(json), 'error');
      }
    } catch {
      toast('Cannot connect to backend. Make sure uvicorn is running on port 8000.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setCustomerState(initialCustomer);
    setTransactionState(initialTransaction);
    setCErrors({});
    setTErrors({});
    setResult(null);
    toast('Form cleared.', 'info');
  };

  const fraudScore = result ? Object.values(result.votes).reduce((s, v) => s + v, 0) : 0;
  const fraudPct   = result ? (fraudScore / 3) * 100 : 0;
  const isFraud    = result?.is_fraud ?? false;

  const chartData = {
    labels: ['Risk', 'Safe'],
    datasets: [{ data: [fraudPct, 100 - fraudPct], backgroundColor: ['#ef4444', '#22c55e'], borderWidth: 0 }],
  };

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>

        {/* ── Form columns ── */}
        <div style={{ flex: 2, display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', gap: '16px' }}>

            {/* Customer Info */}
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>Customer Info</h3>
              <Field label="Customer ID" error={cErrors.customerId}>
                <input style={inputStyle(!!cErrors.customerId)} placeholder="e.g. ACC_00001"
                  value={customer.customerId}
                  onChange={e => onCustomerChange('customerId', e.target.value)} />
              </Field>
              <Field label="Date of Birth" error={cErrors.dob}>
                <input style={inputStyle(!!cErrors.dob)} type="date"
                  value={customer.dob}
                  onChange={e => onCustomerChange('dob', e.target.value)} />
              </Field>
              <Field label="Gender">
                <select style={inputStyle()} value={customer.gender}
                  onChange={e => onCustomerChange('gender', e.target.value)}>
                  {GENDERS.map(g => <option key={g}>{g}</option>)}
                </select>
              </Field>
              <Field label="Location" error={cErrors.location}>
                <input style={inputStyle(!!cErrors.location)} placeholder="e.g. Ho Chi Minh"
                  value={customer.location}
                  onChange={e => onCustomerChange('location', e.target.value)} />
              </Field>
              <Field label="Working Status">
                <select style={inputStyle()} value={customer.workingStatus}
                  onChange={e => onCustomerChange('workingStatus', e.target.value)}>
                  {WORKING_STATUSES.map(s => <option key={s}>{s}</option>)}
                </select>
              </Field>
              <Field label="Monthly Salary (VND)" error={cErrors.salary}>
                <input style={inputStyle(!!cErrors.salary)} type="number" placeholder="e.g. 20000000"
                  value={customer.salary}
                  onChange={e => onCustomerChange('salary', e.target.value)} />
              </Field>
            </div>

            {/* Transaction Info */}
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>Transaction Info</h3>
              <Field label="Transaction ID" error={tErrors.transactionId}>
                <input style={inputStyle(!!tErrors.transactionId)} placeholder="e.g. TXN_00001"
                  value={transaction.transactionId}
                  onChange={e => onTransactionChange('transactionId', e.target.value)} />
              </Field>
              <Field label="Sender Account ID">
                <input style={inputStyle()} placeholder="Optional"
                  value={transaction.senderId}
                  onChange={e => onTransactionChange('senderId', e.target.value)} />
              </Field>
              <Field label="Receiver Account ID">
                <input style={inputStyle()} placeholder="Optional"
                  value={transaction.receiverId}
                  onChange={e => onTransactionChange('receiverId', e.target.value)} />
              </Field>
              <Field label="Amount (VND)" error={tErrors.amount}>
                <input style={inputStyle(!!tErrors.amount)} type="number" placeholder="e.g. 5000000"
                  value={transaction.amount}
                  onChange={e => onTransactionChange('amount', e.target.value)} />
              </Field>
              <Field label="Account Balance (VND)" error={tErrors.accountBalance}>
                <input style={inputStyle(!!tErrors.accountBalance)} type="number" placeholder="e.g. 100000000"
                  value={transaction.accountBalance}
                  onChange={e => onTransactionChange('accountBalance', e.target.value)} />
              </Field>
              <Field label="Date">
                <input style={inputStyle()} type="date" value={transaction.date}
                  onChange={e => onTransactionChange('date', e.target.value)} />
              </Field>
              <Field label="Time (HH : MM)" error={tErrors.hour || tErrors.minute}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input style={{ ...inputStyle(!!tErrors.hour), width: 64, marginBottom: 0 }} type="number"
                    min="0" max="23" placeholder="HH"
                    value={transaction.hour}
                    onChange={e => onTransactionChange('hour', e.target.value)} />
                  <span>:</span>
                  <input style={{ ...inputStyle(!!tErrors.minute), width: 64, marginBottom: 0 }} type="number"
                    min="0" max="59" placeholder="MM"
                    value={transaction.minute}
                    onChange={e => onTransactionChange('minute', e.target.value)} />
                </div>
              </Field>
              <Field label="Device">
                <select style={inputStyle()} value={transaction.device}
                  onChange={e => onTransactionChange('device', e.target.value)}>
                  {DEVICES.map(d => <option key={d}>{d}</option>)}
                </select>
              </Field>
              <Field label="Transaction Detail" error={tErrors.detail}>
                <input style={inputStyle(!!tErrors.detail)} placeholder="e.g. Monthly Salary"
                  value={transaction.detail}
                  onChange={e => onTransactionChange('detail', e.target.value)} />
              </Field>

            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '12px' }}>
            <button id="btn-predict" onClick={handlePredict} disabled={isLoading} style={primaryBtnStyle}>
              {isLoading ? ' Processing…' : 'Run AI Prediction'}
            </button>
            <button id="btn-reset" onClick={handleReset} style={secondaryBtnStyle}>Reset</button>
          </div>
        </div>

        {/* ── Result Panel ── */}
        <div style={resultPanelStyle}>
          <h3 style={{ marginBottom: '16px', fontWeight: 700, color: '#1f2937' }}>AI Risk Assessment</h3>
          <div style={{ position: 'relative', width: '180px', height: '180px', margin: '0 auto' }}>
            <Doughnut data={chartData} options={{ plugins: { legend: { display: false } }, maintainAspectRatio: false }} />
            <div style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
              fontSize: '1.8rem', fontWeight: 800,
              color: fraudPct > 50 ? '#ef4444' : '#22c55e',
            }}>
              {fraudPct.toFixed(0)}%
            </div>
          </div>

          {result === null ? (
            <p style={{ color: '#9ca3af', textAlign: 'center', marginTop: '16px' }}>
              Fill the form and click Predict
            </p>
          ) : (
            <div style={{ marginTop: '16px' }}>
              <div style={{
                padding: '14px', borderRadius: '10px', textAlign: 'center',
                fontWeight: 700, fontSize: '1rem', color: '#fff',
                backgroundColor: isFraud ? '#ef4444' : '#22c55e',
              }}>
                {isFraud ? ' HIGH RISK DETECTED' : 'TRANSACTION IS SAFE'}
              </div>

              <div style={{ marginTop: '14px' }}>
                <p style={labelStyle}>Model Votes ({fraudScore}/3 flagged)</p>
                {Object.entries(result.votes).map(([model, vote]) => (
                  <div key={model} style={voteRowStyle}>
                    <span style={{ fontSize: '0.82rem', color: '#374151', textTransform: 'capitalize' }}>
                      {model.replace(/_/g, ' ')}
                    </span>
                    <span style={{ fontWeight: 700, color: vote ? '#ef4444' : '#22c55e' }}>
                      {vote ? 'FRAUD' : 'SAFE'}
                    </span>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: '14px', padding: '12px', background: '#f9fafb', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
                <p style={labelStyle}>Regression: Predicted Normal Amount</p>
                <p style={{ fontSize: '1.2rem', fontWeight: 700, color: '#2563eb' }}>
                  {result.predicted_amount.toLocaleString('vi-VN')} VND
                </p>
                <p style={{ fontSize: '0.78rem', color: '#6b7280', marginTop: '4px' }}>
                  Actual: {Number(transaction.amount).toLocaleString('vi-VN')} VND
                  {Number(transaction.amount) > result.predicted_amount * 3 && (
                    <span style={{ color: '#ef4444', marginLeft: '6px' }}>⚠ &gt;3× predicted</span>
                  )}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Field wrapper with label + inline error ────────────────────────────────
const Field = ({ label, error, children }) => (
  <div style={{ marginBottom: error ? 4 : 8 }}>
    <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>
      {label}
    </label>
    {children}
    {error && <p style={{ color: '#ef4444', fontSize: '0.75rem', marginTop: 3, marginBottom: 4 }}>{error}</p>}
  </div>
);

// ── Styles ────────────────────────────────────────────────────────────────
const cardStyle = {
  flex: 1, background: '#fff', padding: '20px',
  borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
  border: '1px solid #e5e7eb',
};
const cardTitleStyle = { marginBottom: '12px', fontWeight: 700, color: '#111827' };
// returns red border if hasError
const inputStyle = (hasError = false) => ({
  width: '100%', padding: '9px 12px', marginBottom: 0,
  borderRadius: '8px',
  border: `1px solid ${hasError ? '#ef4444' : '#d1d5db'}`,
  boxShadow: hasError ? '0 0 0 2px rgba(239,68,68,0.15)' : 'none',
  fontSize: '0.88rem', outline: 'none', display: 'block',
  background: hasError ? '#fff5f5' : '#fff',
});
const primaryBtnStyle = {
  flex: 2, padding: '13px', background: '#1d4ed8', color: '#fff',
  border: 'none', borderRadius: '8px', fontWeight: 700,
  cursor: 'pointer', fontSize: '0.95rem',
};
const secondaryBtnStyle = {
  flex: 1, padding: '13px', background: '#f3f4f6', color: '#374151',
  border: '1px solid #d1d5db', borderRadius: '8px', fontWeight: 600, cursor: 'pointer',
};
const resultPanelStyle = {
  flex: 1, background: '#fff', padding: '24px',
  borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
  border: '1px solid #e5e7eb',
};
const labelStyle  = { fontSize: '0.78rem', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', marginBottom: '6px' };
const voteRowStyle = { display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f3f4f6' };

export default Form;