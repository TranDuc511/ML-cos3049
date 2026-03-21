import React, { useState, useMemo } from 'react'

const Customers = () => {
  // --- 1. DATA AND SEARCH ---
  
  const [search, setSearch] = useState('')

  // Put your real customer data in this empty list []
  const customerData = []

  // This handles the Search filtering
  const filteredData = useMemo(() => {
    const keyword = search.toLowerCase().trim()
    return customerData.filter(person =>
      person.id.toLowerCase().includes(keyword)
    )
  }, [search, customerData])

  // Helper to show $ values
  const formatMoney = (value) => {
    return `$${value.toLocaleString('en-US')}`
  }

  // --- 2. THE FACE OF THE PAGE ---
  return (
    <div className="view-customers">
      
      {/* Search Input */}
      <div className="search-bar-wrap">
        <input
          type="text"
          className="search-input"
          placeholder="Search for a Customer ID..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="content-panel">
        <h2 className="viz-title">2. Customer Table</h2>
        
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Birthday</th>
                <th>Gender</th>
                <th>Location</th>
                <th>Balance</th>
                <th>Last Active</th>
                <th>Status</th>
                <th>Monthly Salary</th>
              </tr>
            </thead>
            <tbody>
              {/* This loop draws the table rows */}
              {filteredData.map(person => (
                <tr key={person.id}>
                  <td>{person.id}</td>
                  <td>{person.dob}</td>
                  <td>{person.gender}</td>
                  <td>{person.location}</td>
                  <td>{formatMoney(person.balance)}</td>
                  <td>{person.transactionTime}</td>
                  <td>{person.workingStatus}</td>
                  <td>{formatMoney(person.salary)}</td>
                </tr>
              ))}

              {filteredData.length === 0 && (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '3rem', color: '#888' }}>
                    No customers found. (Ready for your real data!)
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

export default Customers
