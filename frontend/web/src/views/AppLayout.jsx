import React, { useState } from 'react'
import Sidebar from '../components/Sidebar'
import Dashboard from './Dashboard'
import TransactionList from './TransactionList'
import Customers from './Customers'
import UploadData from './UploadData'

/**
 * AppLayout Component
 * This is the main shell that holds the Sidebar and shows the selected page.
 */
const AppLayout = ({ onLogout }) => {
  // --- 1. SETTINGS AND STATE ---
  
  // 'currentView' keeps track of which page we are looking at (default is 'dashboard')
  const [currentView, setCurrentView] = useState('dashboard') 
  
  // 'isDrawerOpen' tracks if the side menu is visible on mobile phones
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)     

  // --- 2. ACTIONS (Functions) ---

  // Opens or closes the mobile side menu
  const toggleDrawer = () => setIsDrawerOpen(!isDrawerOpen)

  // This runs when you click a link in the sidebar
  const handleViewChange = (view) => {
    setCurrentView(view)      // Change the page
    setIsDrawerOpen(false)    // Close the side menu (if on mobile)
  }

  // Figure out the title of the current page
  const getPageTitle = () => {
    if (currentView === 'dashboard') return 'Dashboard'
    if (currentView === 'transactions') return 'Transaction List'
    if (currentView === 'customers') return 'Customer Segmentation'
    if (currentView === 'upload') return 'Upload Data'
    if (currentView === 'accounts') return 'Account List'
    if (currentView === 'report') return 'AI Report'
    return 'Dashboard'
  }

  // --- 3. THE FACE OF THE WEB (HTML/JSX) ---
  return (
    <div className="app">
      {/* This background dims the screen when the mobile menu is open */}
      <div className={`overlay ${isDrawerOpen ? 'active' : ''}`} onClick={toggleDrawer}></div>
      
      {/* The Side Menu Navigation */}
      <Sidebar 
        currentView={currentView} 
        onViewChange={handleViewChange} 
        onLogout={onLogout}
        breadcrumb={getPageTitle()}
        isOpen={isDrawerOpen}
        onToggle={toggleDrawer}
      />

      <main className="main">
        {/* Mobile Header: Only shows up on mobile screens */}
        <header className="mobile-header">
          <button className="menu-toggle" onClick={toggleDrawer}>☰</button>
          <span className="account-name">{getPageTitle()}</span>
        </header>

        {/* Main Content Area: Change this to swap what is shown on screen */}
        <section className={`view view-${currentView}`}>
          <h1 className="page-title">Anomaly Transaction Detection</h1>
          
          {/* We show the correct component based on 'currentView' */}
          {currentView === 'dashboard' && <Dashboard />}
          {currentView === 'transactions' && <TransactionList />}
          {currentView === 'customers' && <Customers />}
          {currentView === 'upload' && <UploadData />}
          
          {/* Simple placeholder pages */}
          {currentView === 'accounts' && (
            <div className="content-panel">
              <p className="placeholder-text">Account List — content area</p>
            </div>
          )}
          {currentView === 'report' && (
            <div className="content-panel">
              <p className="placeholder-text">AI Report — content area</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default AppLayout
