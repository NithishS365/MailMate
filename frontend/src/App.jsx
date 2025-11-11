import React, { useState, useEffect } from 'react';
import { emailService } from './services/api';
import EmailList from './components/EmailList';
import Statistics from './components/Statistics';
import DailyDigest from './components/DailyDigest';
import AdvancedSearch from './components/AdvancedSearch';
import { 
  FaEnvelope, 
  FaBrain, 
  FaChartLine, 
  FaInbox, 
  FaStar, 
  FaHome,
  FaBookmark,
  FaCog,
  FaBars,
  FaTimes
} from 'react-icons/fa';
import axios from 'axios';
import './styles/App.css';

function App() {
  const [emails, setEmails] = useState([]);
  const [filteredEmails, setFilteredEmails] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [activeFilters, setActiveFilters] = useState({});
  const [view, setView] = useState('all'); // 'all', 'unread', 'priority'
  const [currentPage, setCurrentPage] = useState('inbox'); // 'inbox', 'analytics', 'digest'
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    loadEmails();
    loadStatistics();
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      loadEmails();
      loadStatistics();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    applyFilters();
  }, [emails, activeFilters, view]);

  const loadEmails = async () => {
    try {
      setLoading(true);
      const data = await emailService.getEmails({ limit: 100 });
      setEmails(data);
      setFilteredEmails(data);
      setError(null);
    } catch (err) {
      setError('Failed to load emails. Please check if the backend server is running.');
      console.error('Error loading emails:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const data = await emailService.getStatistics();
      setStatistics(data);
    } catch (err) {
      console.error('Error loading statistics:', err);
    }
  };

  const handleEmailClick = async (email) => {
    if (email.is_read === 0) {
      try {
        await emailService.updateEmail(email.id, { is_read: 1 });
        await loadEmails();
        await loadStatistics();
      } catch (err) {
        console.error('Error updating email:', err);
      }
    }
  };

  const applyFilters = () => {
    let filtered = [...emails];

    // Apply view filter
    if (view === 'unread') {
      filtered = filtered.filter((email) => email.is_read === 0);
    } else if (view === 'priority') {
      filtered = filtered.filter((email) => email.priority === 'HIGH');
    }

    // Apply search filter
    if (activeFilters.search) {
      const searchLower = activeFilters.search.toLowerCase();
      filtered = filtered.filter(
        (email) =>
          email.subject.toLowerCase().includes(searchLower) ||
          email.sender.toLowerCase().includes(searchLower) ||
          email.body.toLowerCase().includes(searchLower)
      );
    }

    // Apply category filter
    if (activeFilters.category) {
      filtered = filtered.filter((email) => email.category === activeFilters.category);
    }

    // Apply priority filter
    if (activeFilters.priority) {
      filtered = filtered.filter((email) => email.priority === activeFilters.priority);
    }

    setFilteredEmails(filtered);
  };

  const handleFilterChange = (filters) => {
    setActiveFilters(filters);
  };

  const handleAdvancedSearch = async (filters) => {
    try {
      setLoading(true);
      // Build query string
      const params = new URLSearchParams();
      if (filters.q) params.append('q', filters.q);
      if (filters.categories.length > 0) {
        filters.categories.forEach(cat => params.append('categories', cat));
      }
      if (filters.priorities.length > 0) {
        filters.priorities.forEach(pri => params.append('priorities', pri));
      }
      if (filters.is_read !== null) params.append('is_read', filters.is_read);
      if (filters.start_date) params.append('start_date', filters.start_date);
      if (filters.end_date) params.append('end_date', filters.end_date);
      if (filters.min_priority_score) params.append('min_priority_score', filters.min_priority_score);
      
      const response = await axios.get(`http://localhost:8000/api/emails/search?${params.toString()}`);
      setEmails(response.data);
      setFilteredEmails(response.data);
      setMessage(`Found ${response.data.length} emails`);
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      setError('Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearSearch = () => {
    loadEmails();
  };

  const handleDeleteEmail = async (emailId) => {
    try {
      await axios.delete(`http://localhost:8000/api/emails/${emailId}`);
      setEmails(emails.filter(e => e.id !== emailId));
      setMessage('Email deleted successfully');
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      setError('Failed to delete email');
      console.error('Delete error:', err);
    }
  };

  const handleToggleStar = (emailId) => {
    // Star functionality (can be extended to backend)
    console.log('Toggle star for email:', emailId);
  };

  const handleArchive = (emailId) => {
    // Archive functionality
    console.log('Archive email:', emailId);
    setMessage('Email archived');
    setTimeout(() => setMessage(null), 2000);
  };

  const renderPageContent = () => {
    if (loading && emails.length === 0) {
      return (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading your emails...</p>
        </div>
      );
    }

    switch (currentPage) {
      case 'inbox':
        return (
          <div className="page-content">
            <div className="page-header">
              <div>
                <h1><FaInbox /> Inbox</h1>
                <p className="page-subtitle">{filteredEmails.length} emails</p>
              </div>
              <div className="view-tabs">
                <button 
                  className={`tab ${view === 'all' ? 'active' : ''}`}
                  onClick={() => setView('all')}
                >
                  All
                </button>
                <button 
                  className={`tab ${view === 'unread' ? 'active' : ''}`}
                  onClick={() => setView('unread')}
                >
                  Unread
                </button>
                <button 
                  className={`tab ${view === 'priority' ? 'active' : ''}`}
                  onClick={() => setView('priority')}
                >
                  Priority
                </button>
              </div>
            </div>
            
            {/* Quick Stats */}
            {statistics && (
              <div className="quick-stats-inline">
                <div className="stat-box" onClick={() => setView('all')}>
                  <FaInbox className="stat-icon" />
                  <div>
                    <div className="stat-number">{statistics.total_emails || 0}</div>
                    <div className="stat-text">Total</div>
                  </div>
                </div>
                <div className="stat-box" onClick={() => setView('unread')}>
                  <FaEnvelope className="stat-icon" />
                  <div>
                    <div className="stat-number">{statistics.unread_count || 0}</div>
                    <div className="stat-text">Unread</div>
                  </div>
                </div>
                <div className="stat-box" onClick={() => setView('priority')}>
                  <FaStar className="stat-icon priority" />
                  <div>
                    <div className="stat-number">{statistics.by_priority?.HIGH || 0}</div>
                    <div className="stat-text">Priority</div>
                  </div>
                </div>
              </div>
            )}

            <div className="card">
              <AdvancedSearch onSearch={handleAdvancedSearch} onClear={handleClearSearch} />
              <EmailList 
                emails={filteredEmails} 
                onEmailClick={handleEmailClick}
                onDelete={handleDeleteEmail}
                onToggleStar={handleToggleStar}
                onArchive={handleArchive}
              />
            </div>
          </div>
        );

      case 'analytics':
        return (
          <div className="page-content">
            <div className="page-header">
              <div>
                <h1><FaChartLine /> Advanced Analytics</h1>
                <p className="page-subtitle">Comprehensive email insights and statistics</p>
              </div>
            </div>
            <div className="card">
              <Statistics />
            </div>
          </div>
        );

      case 'digest':
        return (
          <div className="page-content">
            <div className="page-header">
              <div>
                <h1><FaBookmark /> Daily Digest</h1>
                <p className="page-subtitle">Your top priority emails</p>
              </div>
            </div>
            <div className="card">
              <DailyDigest />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className={`sidebar-nav ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo">
            <FaEnvelope className="logo-icon" />
            {sidebarOpen && <span className="logo-text">MailMate</span>}
          </div>
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <FaTimes /> : <FaBars />}
          </button>
        </div>

        <nav className="sidebar-menu">
          <button 
            className={`menu-item ${currentPage === 'inbox' ? 'active' : ''}`}
            onClick={() => setCurrentPage('inbox')}
          >
            <FaInbox className="menu-icon" />
            {sidebarOpen && <span>Inbox</span>}
            {sidebarOpen && statistics && (
              <span className="badge-count">{statistics.unread_count || 0}</span>
            )}
          </button>

          <button 
            className={`menu-item ${currentPage === 'analytics' ? 'active' : ''}`}
            onClick={() => setCurrentPage('analytics')}
          >
            <FaChartLine className="menu-icon" />
            {sidebarOpen && <span>Analytics</span>}
          </button>

          <button 
            className={`menu-item ${currentPage === 'digest' ? 'active' : ''}`}
            onClick={() => setCurrentPage('digest')}
          >
            <FaBookmark className="menu-icon" />
            {sidebarOpen && <span>Daily Digest</span>}
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-info">
            {sidebarOpen && (
              <div style={{ padding: '16px 20px', fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)' }}>
                <p>AI-Powered Email System</p>
                <p>v1.0.0</p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-area">
        <header className="top-header">
          <div className="header-left">
            <button className="mobile-menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
              <FaBars />
            </button>
            <h2>AI-Powered Email Management</h2>
          </div>
          <div className="header-actions">
            <button 
              className="btn btn-icon" 
              onClick={() => { loadEmails(); loadStatistics(); }}
              disabled={loading}
              title="Refresh"
            >
              <FaChartLine />
            </button>
          </div>
        </header>

        {message && <div className="notification success">{message}</div>}
        {error && <div className="notification error">{error}</div>}

        <div className="content-wrapper">
          {renderPageContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
