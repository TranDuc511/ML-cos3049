import { useState } from 'react'
import LoginView from './views/LoginView'
import AppLayout from './views/AppLayout'

function App() {
  // --- 1. SETTINGS ---
  // This 'isLoggedIn' variable is like a switch. 
  // false = Show Login Screen, true = Show Dashboard
  const [isLoggedIn, setIsLoggedIn] = useState(false)

  // --- 2. ACTIONS ---
  
  // This runs when the user clicks 'Login'
  const handleLogin = (e) => {
    e.preventDefault()
    setIsLoggedIn(true) // Flip the switch to 'true'
  }

  // This runs when the user clicks 'Logout'
  const handleLogout = () => {
    setIsLoggedIn(false) // Flip the switch back to 'false'
  }

  // --- 3. WHAT THE USER SEES ---
  return (
    <div className="app-container">
      {/* If logged in, show the Dashboard. Otherwise, show the Login page. */}
      {isLoggedIn ? (
        <AppLayout onLogout={handleLogout} />
      ) : (
        <LoginView onLogin={handleLogin} />
      )}
    </div>
  )
}

export default App
