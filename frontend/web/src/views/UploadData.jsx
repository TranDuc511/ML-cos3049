import React, { useState, useRef } from 'react'

/**
 * UploadData View
 * Provides an interactive drag-and-drop interface for uploading files.
 */
const UploadData = () => {
  // --- State ---
  const [fileNames, setFileNames] = useState('')
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null) // Reference to the hidden <input type="file" />

  /**
   * Processes the files selected or dropped.
   * Update this function to actually upload files to a server.
   */
  const handleFiles = (files) => {
    if (files && files.length > 0) {
      const names = Array.from(files).map(f => f.name).join(', ')
      setFileNames(names)
    } else {
      setFileNames('')
    }
  }

  // --- Drag & Drop Handlers ---
  const onDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const onDragLeave = () => {
    setIsDragOver(false)
  }

  return (
    <div className="view-upload">
      <div className="content-panel">
        <h2 className="viz-title">Upload Data File</h2>
        <div className="upload-wrap">
          <input 
            type="file" 
            className="upload-input" 
            multiple 
            ref={fileInputRef}
            onChange={(e) => handleFiles(e.target.files)}
            style={{ display: 'none' }}
          />
          <div 
            className={`upload-dropzone ${isDragOver ? 'dragover' : ''}`}
            onClick={() => fileInputRef.current.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <div className="upload-icon">⬆</div>
            <p className="upload-title">Drop your file here</p>
            <p className="upload-subtitle">or click to browse from your computer</p>
          </div>
          <div className="upload-file-name">
            {fileNames && `Selected: ${fileNames}`}
          </div>
        </div>
      </div>
    </div>
  )
}

export default UploadData
