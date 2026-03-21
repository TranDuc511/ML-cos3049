import React from 'react'

const LoginView = ({ onLogin }) => {
  return (
    <main className="login-card">
      <h1 className="login-title">Sign in</h1>
      <p className="login-subtitle">Access the Underground Transaction Detection dashboard.</p>
      <form id="login-form" onSubmit={onLogin}>
        <div className="login-field">
          <label className="login-label" htmlFor="login-username">Username</label>
          <input id="login-username" className="login-input" type="text" autoComplete="username" required />
        </div>
        <div className="login-field">
          <label className="login-label" htmlFor="login-password">Password</label>
          <input id="login-password" className="login-input" type="password" autoComplete="current-password" required />
        </div>
        <button type="submit" className="login-button">Login</button>
      </form>
      <style>{`
        body {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background: radial-gradient(circle at top left, #1d4ed8 0, transparent 55%), var(--main-bg);
          margin: 0;
        }
      `}</style>
    </main>
  )
}

export default LoginView
