import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import LoginView from './views/LoginView'
import AppLayout from './views/AppLayout'
import { ToastProvider } from './components/Toast'

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() =>
    sessionStorage.getItem('isLoggedIn') === 'true'
  )

  const handleLogin = () => {
    sessionStorage.setItem('isLoggedIn', 'true')
    setIsLoggedIn(true)
  }

  const handleLogout = () => {
    sessionStorage.removeItem('isLoggedIn')
    setIsLoggedIn(false)
  }

  return (
    <ToastProvider>
      <Router>
        <div className="app-container">
          <Routes>
            <Route
              path="/login"
              element={!isLoggedIn ? <LoginView onLogin={handleLogin} /> : <Navigate to="/" />}
            />
            <Route
              path="/*"
              element={isLoggedIn ? <AppLayout onLogout={handleLogout} /> : <Navigate to="/login" />}
            />
          </Routes>
        </div>
      </Router>
    </ToastProvider>
  )
}

export default App