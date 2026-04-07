// Toast.jsx — Lightweight toast notification system
import React, { createContext, useContext, useState, useCallback } from 'react';

const ToastCtx = createContext(null);

let _id = 0;

export const ToastProvider = ({ children }) => {
  const [toasts, setToasts] = useState([]);

  const add = useCallback((message, type = 'info', duration = 3500) => {
    const id = ++_id;
    setToasts(t => [...t, { id, message, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), duration);
  }, []);

  return (
    <ToastCtx.Provider value={add}>
      {children}
      <div style={containerStyle}>
        {toasts.map(t => (
          <div key={t.id} style={{ ...toastStyle, ...typeStyles[t.type] }}>
            <span style={{ marginRight: 8 }}>{icons[t.type]}</span>
            {t.message}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
};

export const useToast = () => {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error('useToast must be inside <ToastProvider>');
  return ctx;
};

const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };

const typeStyles = {
  success: { borderLeft: '4px solid #22c55e', background: '#f0fdf4' },
  error:   { borderLeft: '4px solid #ef4444', background: '#fef2f2' },
  warning: { borderLeft: '4px solid #f59e0b', background: '#fffbeb' },
  info:    { borderLeft: '4px solid #3b82f6', background: '#eff6ff' },
};

const containerStyle = {
  position: 'fixed', bottom: 24, right: 24,
  display: 'flex', flexDirection: 'column', gap: 10,
  zIndex: 9999, maxWidth: 360,
};

const toastStyle = {
  padding: '12px 16px', borderRadius: 10,
  boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
  fontSize: '0.9rem', color: '#1f2937', fontWeight: 500,
  display: 'flex', alignItems: 'center',
  animation: 'slideIn 0.25s ease',
};
