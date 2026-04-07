import React from 'react'

/**
 * Sidebar Component
 * This is the menu on the left side of the screen.
 * It also becomes a "drawer" menu on mobile phones.
 */
const Sidebar = ({ currentView, onViewChange, onLogout, breadcrumb, isOpen, onToggle }) => {
  
  /**
   * List of menu items. 
   * 'key' is used in the code, 'label' is what the user sees.
   */
  const navItems = [
    { key: 'dashboard', label: 'Dashboard' },
    { key: 'transactions', label: 'Transaction List' },
    { key: 'form', label: 'Predict Fraud' },
  ]


  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      {/* 1. TOP SECTION: Account name and breadcrumb */}
      <div className="sidebar-account">
        <div className="sidebar-breadcrumb">{breadcrumb}</div>
        
        {/* This 'X' button only shows up on mobile when the menu is open */}
        <button className="menu-toggle" onClick={onToggle}>✕</button>
        
        <div className="avatar"></div>
        <span className="account-name">Account Name</span>
      </div>

      {/* 2. MIDDLE SECTION: Navigation Links */}
      <nav className="sidebar-nav">
        <h3 className="nav-heading">Overview</h3>
        <ul className="nav-links">
          {navItems.map(item => (
            <li key={item.key}>
              <a 
                href="#" 
                className={`nav-link ${currentView === item.key ? 'active' : ''}`}
                onClick={(e) => {
                  e.preventDefault()        // Prevent page refresh
                  onViewChange(item.key)    // Tell AppLayout to swap the page
                }}
              >
                → {item.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      {/* 3. BOTTOM SECTION: Logout Button */}
      <div className="sidebar-footer">
        <a href="#" className="logout" onClick={(e) => {
          e.preventDefault()
          onLogout()
        }}>
          Logout
        </a>
      </div>

      {/* Small design rule: Ensure the mobile close button stays white and visible */}
      <style>{`
        .sidebar .menu-toggle {
          color: white;
          position: absolute;
          right: 15px;
          top: 15px;
          display: none;
        }
        @media (max-width: 900px) {
          .sidebar .menu-toggle { display: block; }
        }
      `}</style>
    </aside>
  )
}

export default Sidebar