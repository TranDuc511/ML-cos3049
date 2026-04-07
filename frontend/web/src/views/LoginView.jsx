import React, { useState } from 'react';

// Login setting 
const LoginView = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    // call App.jsx to confirm log in successful
    // testing login
    onLogin(); 
  };

  return (
    <div style={containerStyle}>
      <div style={loginCardStyle}>
        <div style={headerSection}>
          <h1 style={titleStyle}>Welcome Back</h1>
          <p style={subtitleStyle}>Access the Underground Transaction Detection dashboard.</p>
        </div>

        <form onSubmit={handleLogin} style={formStyle}>
          <div style={inputGroup}>
            <label style={labelStyle}>Username</label>
            <input 
              type="text" 
              placeholder="Enter your username" 
              style={inputFieldStyle}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>

          <div style={inputGroup}>
            <label style={labelStyle}>Password</label>
            <input 
              type="password" 
              placeholder="••••••••" 
              style={inputFieldStyle}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button type="submit" style={loginButtonStyle}>
            Sign In
          </button>
        </form>
      </div>
    </div>
  );
};

// --- CSS STYLES ---
const containerStyle = {
  height: '100vh', width: '100vw', display: 'flex',
  justifyContent: 'center', alignItems: 'center',
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  fontFamily: "'Segoe UI', sans-serif",
};

const loginCardStyle = {
  backgroundColor: '#ffffff', padding: '40px', borderRadius: '15px',
  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)', width: '100%', maxWidth: '400px',
};

const headerSection = { marginBottom: '30px', textAlign: 'center' };
const titleStyle = { margin: '0', fontSize: '2rem', color: '#2d3436' };
const subtitleStyle = { color: '#636e72', fontSize: '0.9rem', marginTop: '10px' };
const formStyle = { display: 'flex', flexDirection: 'column', gap: '20px' };
const inputGroup = { textAlign: 'left' };
const labelStyle = { display: 'block', marginBottom: '8px', fontWeight: 'bold', color: '#2d3436', fontSize: '0.85rem' };

const inputFieldStyle = {
  width: '100%', padding: '12px 15px', borderRadius: '8px',
  border: '1px solid #dfe6e9', outline: 'none', fontSize: '1rem', boxSizing: 'border-box',
};

const loginButtonStyle = {
  backgroundColor: '#6c5ce7', color: 'white', padding: '12px',
  borderRadius: '8px', border: 'none', fontSize: '1rem',
  fontWeight: 'bold', cursor: 'pointer', marginTop: '10px',
};

export default LoginView;