(function () {
  const views = {
    overview: { id: 'view-overview', label: 'Overview' },
    transactions: { id: 'view-transactions', label: 'Transaction List' },
    customers: { id: 'view-customers', label: 'Customer Segmentation' },
    accounts: { id: 'view-accounts', label: 'Account List' },
    report: { id: 'view-report', label: 'AI Report' },
    upload: { id: 'view-upload', label: 'Upload Data' }
  };

  const breadcrumbEl = document.getElementById('sidebar-breadcrumb');
  const navLinks = document.querySelectorAll('.nav-link[data-view]');

  // ===== Mock data layer (ready for API/database integration) =====
  // Transaction dataset schema follow table form:
  // Transaction ID, Sender Account ID, Receiver Account ID, Transaction amount,
  // Timestamp, Transaction Detail, Geological, Device Use.
  // Initialized as empty; to be populated with data from API or database.
  const transactionData = [];

  const customerData = [
    {
      id: 'CUST-001',
      dob: '1990-01-12',
      gender: 'Female',
      location: 'Ho Chi Minh City',
      balance: 12500,
      transactionTime: '09:15',
      workingStatus: 'Employed',
      salary: 1200,
    },
    {
      id: 'CUST-002',
      dob: '1987-03-22',
      gender: 'Male',
      location: 'Hanoi',
      balance: 8300,
      transactionTime: '13:40',
      workingStatus: 'Employed',
      salary: 900,
    },
    {
      id: 'CUST-003',
      dob: '1995-07-05',
      gender: 'Female',
      location: 'Da Nang',
      balance: 3750,
      transactionTime: '18:25',
      workingStatus: 'Unemployed',
      salary: 400,
    },
  ];

  // ===== Rendering layer (chỉ phụ trách vẽ UI từ data) =====
  const transactionsTbody = document.getElementById('transactions-tbody');
  const customersTbody = document.getElementById('customers-tbody');

  function formatCurrencyUSD(value) {
    return `$${value.toLocaleString('en-US')}`;
  }

  function renderTransactions(rows) {
    if (!transactionsTbody) return;
    transactionsTbody.innerHTML = rows
      .map(t => {
        return `
          <tr>
            <td>${t.id}</td>
            <td>${t.senderAccountId}</td>
            <td>${t.receiverAccountId}</td>
            <td>${typeof t.amount === 'number' ? formatCurrencyUSD(t.amount) : (t.amount || '')}</td>
            <td>${t.timestamp || ''}</td>
            <td>${t.detail || ''}</td>
            <td>${t.geo || ''}</td>
            <td>${t.deviceUse || ''}</td>
          </tr>
        `;
      })
      .join('');
  }

  function renderCustomers(rows) {
    if (!customersTbody) return;
    customersTbody.innerHTML = rows
      .map(c => {
        return `
          <tr>
            <td>${c.id}</td>
            <td>${c.dob}</td>
            <td>${c.gender}</td>
            <td>${c.location}</td>
            <td>${formatCurrencyUSD(c.balance)}</td>
            <td>${c.transactionTime}</td>
            <td>${c.workingStatus}</td>
            <td>${formatCurrencyUSD(c.salary)}</td>
          </tr>
        `;
      })
      .join('');
  }

  function showView(viewKey) {
    const config = views[viewKey];
    if (!config) return;

    document.querySelectorAll('.view').forEach(el => el.classList.add('hidden'));
    const target = document.getElementById(config.id);
    if (target) target.classList.remove('hidden');

    if (breadcrumbEl) breadcrumbEl.textContent = config.label;

    navLinks.forEach(link => {
      link.classList.toggle('active', link.getAttribute('data-view') === viewKey);
    });
  }

  navLinks.forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const view = this.getAttribute('data-view');
      showView(view);
    });
  });

  // Upload drag & drop / click-to-browse
  const uploadInput = document.getElementById('upload-input');
  const uploadDropzone = document.getElementById('upload-dropzone');
  const uploadFileName = document.getElementById('upload-file-name');

  if (uploadDropzone && uploadInput) {
    uploadDropzone.addEventListener('click', () => {
      uploadInput.click();
    });

    uploadInput.addEventListener('change', () => {
      if (!uploadFileName) return;
      if (uploadInput.files && uploadInput.files.length > 0) {
        const names = Array.from(uploadInput.files).map(f => f.name).join(', ');
        uploadFileName.textContent = `Selected: ${names}`;
      } else {
        uploadFileName.textContent = '';
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadDropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadDropzone.classList.remove('dragover');
      });
    });

    uploadDropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (!dt || !uploadFileName) return;
      const files = dt.files;
      if (files && files.length > 0) {
        const names = Array.from(files).map(f => f.name).join(', ');
        uploadFileName.textContent = `Dropped: ${names}`;
      }
    });
  }

  // ===== Search and filter logic (processes the data layer) =====
  const transactionSearchInput = document.getElementById('transaction-search');
  const customerSearchInput = document.getElementById('customer-search');

  if (transactionSearchInput) {
    transactionSearchInput.addEventListener('input', () => {
      const keyword = transactionSearchInput.value.trim().toLowerCase();
      const filtered = transactionData.filter(t => {
        return (
          (t.id || '').toLowerCase().includes(keyword) ||
          (t.senderAccountId || '').toLowerCase().includes(keyword) ||
          (t.receiverAccountId || '').toLowerCase().includes(keyword)
        );
      });
      renderTransactions(filtered);
    });
  }

  if (customerSearchInput) {
    customerSearchInput.addEventListener('input', () => {
      const keyword = customerSearchInput.value.trim().toLowerCase();
      const filtered = customerData.filter(c =>
        c.id.toLowerCase().includes(keyword)
      );
      renderCustomers(filtered);
    });
  }

  // Initial data rendering (replace with API-driven data in production)
  renderTransactions(transactionData);
  renderCustomers(customerData);

  showView('overview');
})();
