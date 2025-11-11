import { useState, useEffect } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';
import axios from 'axios';
import '../styles/Statistics.css';

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1', '#d084d1', '#a4de6c'];

const Statistics = () => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadAnalytics();
    const interval = setInterval(loadAnalytics, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/api/analytics/advanced');
      setAnalytics(response.data);
    } catch (error) {
      console.error('Error loading analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="stats-loading">Loading analytics...</div>;
  }

  if (!analytics || analytics.total_emails === 0) {
    return <div className="stats-empty">No data available for analytics</div>;
  }

  const { summary_stats } = analytics;

  return (
    <div className="statistics-container">
      {/* Summary Cards */}
      <div className="stats-summary-cards">
        <div className="stats-card" title="Total number of emails in the system">
          <div className="stats-icon">📧</div>
          <div className="stats-info">
            <h3>{analytics.total_emails}</h3>
            <p>Total Emails</p>
          </div>
        </div>
        <div className="stats-card" title="Average priority score across all emails">
          <div className="stats-icon">⭐</div>
          <div className="stats-info">
            <h3>{summary_stats.avg_priority_score}</h3>
            <p>Avg Priority</p>
          </div>
        </div>
        <div className="stats-card" title="Percentage of emails that have been read">
          <div className="stats-icon">✅</div>
          <div className="stats-info">
            <h3>{summary_stats.read_percentage}%</h3>
            <p>Read Rate</p>
          </div>
        </div>
        <div className="stats-card priority" title="Number of high priority emails">
          <div className="stats-icon">🔥</div>
          <div className="stats-info">
            <h3>{summary_stats.high_priority_count}</h3>
            <p>High Priority</p>
          </div>
        </div>
        <div className="stats-card" title="Number of unique email senders">
          <div className="stats-icon">👥</div>
          <div className="stats-info">
            <h3>{summary_stats.unique_senders}</h3>
            <p>Unique Senders</p>
          </div>
        </div>
        <div className="stats-card" title="Email volume change over the last 7 days">
          <div className="stats-icon">{summary_stats.volume_trend.change_percentage >= 0 ? '📈' : '📉'}</div>
          <div className="stats-info">
            <h3 style={{ color: summary_stats.volume_trend.change_percentage >= 0 ? '#2ed573' : '#ff4757' }}>
              {summary_stats.volume_trend.change_percentage >= 0 ? '+' : ''}{summary_stats.volume_trend.change_percentage}%
            </h3>
            <p>7-Day Trend</p>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="stats-tabs">
        <button 
          className={activeTab === 'overview' ? 'stats-tab active' : 'stats-tab'}
          onClick={() => setActiveTab('overview')}
        >
          📊 Overview
        </button>
        <button 
          className={activeTab === 'time' ? 'stats-tab active' : 'stats-tab'}
          onClick={() => setActiveTab('time')}
        >
          ⏰ Time Analysis
        </button>
        <button 
          className={activeTab === 'senders' ? 'stats-tab active' : 'stats-tab'}
          onClick={() => setActiveTab('senders')}
        >
          👥 Top Senders
        </button>
      </div>

      {/* Tab Content */}
      <div className="stats-content">
        {activeTab === 'overview' && (
          <div className="stats-tab-content">
            <div className="stats-chart-card">
              <h3>📂 Category Distribution by Priority</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.category_priority_breakdown}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="high" stackId="a" fill="#ff4757" name="High" />
                  <Bar dataKey="medium" stackId="a" fill="#ffa502" name="Medium" />
                  <Bar dataKey="low" stackId="a" fill="#2ed573" name="Low" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="stats-chart-card">
              <h3>📖 Read vs Unread by Category</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.category_read_matrix}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="read" fill="#2ed573" name="Read" />
                  <Bar dataKey="unread" fill="#ff6348" name="Unread" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="stats-chart-card">
              <h3>⭐ Priority Score Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={analytics.priority_score_distribution}
                    dataKey="count"
                    nameKey="priority_range"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={(entry) => `${entry.priority_range}: ${entry.count}`}
                  >
                    {analytics.priority_score_distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="stats-chart-card">
              <h3>📏 Email Length Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.email_length_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="length_category" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" name="Emails" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'time' && (
          <div className="stats-tab-content">
            <div className="stats-chart-card full-width">
              <h3>📅 Email Volume Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analytics.time_series}>
                  <defs>
                    <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="count" stroke="#8884d8" fillOpacity={1} fill="url(#colorCount)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="stats-chart-card">
              <h3>🕐 Hourly Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analytics.hourly_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#82ca9d" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="stats-chart-card">
              <h3>📆 Day of Week Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.day_of_week_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day_of_week" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'senders' && (
          <div className="stats-tab-content">
            <div className="stats-chart-card full-width">
              <h3>👤 Top 10 Email Senders</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={analytics.top_senders} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="sender" type="category" width={120} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="total_emails" fill="#8884d8" name="Total" />
                  <Bar dataKey="read_emails" fill="#82ca9d" name="Read" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="stats-chart-card full-width">
              <h3>📊 Sender Statistics Table</h3>
              <div className="stats-table-container">
                <table className="stats-table">
                  <thead>
                    <tr>
                      <th>Sender</th>
                      <th>Total</th>
                      <th>Read</th>
                      <th>Avg Priority</th>
                      <th>Read Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.top_senders.map((sender, idx) => (
                      <tr key={idx}>
                        <td>{sender.sender}</td>
                        <td>{sender.total_emails}</td>
                        <td>{sender.read_emails}</td>
                        <td>{sender.avg_priority.toFixed(3)}</td>
                        <td>
                          <span className={`rate-badge ${((sender.read_emails / sender.total_emails) * 100) > 50 ? 'high' : 'low'}`}>
                            {((sender.read_emails / sender.total_emails) * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Statistics;
