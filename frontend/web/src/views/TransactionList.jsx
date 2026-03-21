import React, { useState, useMemo } from 'react'

const TransactionList = () => {
  // --- 1. DATA AND SEARCH ---
  
  // 'search' stores what the user types in the search box
  const [search, setSearch] = useState('')
  
  // Put your real data in this empty list [] to show it on screen
  const transactionData = [] 

  // This part handles the SEARCH logic
  const filteredData = useMemo(() => {
    const keyword = search.toLowerCase().trim()
    
    return transactionData.filter(item => {
      // Check if ID or Account names match what was typed
      const idMatch = (item.id || '').toLowerCase().includes(keyword)
      const senderMatch = (item.senderAccountId || '').toLowerCase().includes(keyword)
      const receiverMatch = (item.receiverAccountId || '').toLowerCase().includes(keyword)
      
      return idMatch || senderMatch || receiverMatch
    })
  }, [search, transactionData])

  // Helper function to show money as $100.00
  const formatMoney = (value) => {
    return `$${value.toLocaleString('en-US')}`
  }

  // --- 2. THE FACE OF THE PAGE ---
  return (
    <div className="view-transactions">
      
      {/* The Search Box Area */}
      <div className="search-bar-wrap">
        <input
          type="text"
          className="search-input"
          placeholder="Type here to search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="content-panel">
        <h2 className="viz-title">1. Transaction Table</h2>
        
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Sender</th>
                <th>Receiver</th>
                <th>Amount</th>
                <th>Time</th>
                <th>Details</th>
                <th>Location</th>
                <th>Device</th>
              </tr>
            </thead>
            <tbody>
              {/* This loop 'draws' a new row for every transaction we found */}
              {filteredData.map((item, index) => (
                <tr key={item.id || index}>
                  <td>{item.id}</td>
                  <td>{item.senderAccountId}</td>
                  <td>{item.receiverAccountId}</td>
                  <td>{typeof item.amount === 'number' ? formatMoney(item.amount) : item.amount}</td>
                  <td>{item.timestamp}</td>
                  <td>{item.detail}</td>
                  <td>{item.geo}</td>
                  <td>{item.deviceUse}</td>
                </tr>
              ))}
              
              {/* If search finds nothing, show this message */}
              {filteredData.length === 0 && (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '3rem', color: '#888' }}>
                    No data found. (Ready for your real data!)
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default TransactionList
