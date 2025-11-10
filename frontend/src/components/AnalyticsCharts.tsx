import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  PieChart,
  TrendingUp,
  Mail,
  Clock,
  Users,
  Filter,
  Download,
  RefreshCw
} from 'lucide-react';
import Plot from 'react-plotly.js';
import { DashboardStats, EmailFilters } from '../types';
import { staticDashboardStats, staticAnalyticsData } from '../services/staticData';

interface AnalyticsChartsProps {
  className?: string;
  filters?: EmailFilters;
}

interface ChartData {
  categoryDistribution: {
    categories: string[];
    counts: number[];
  };
  priorityDistribution: {
    priorities: string[];
    counts: number[];
  };
  timeSeriesData: {
    dates: string[];
    counts: number[];
  };
  senderAnalytics: {
    senders: string[];
    counts: number[];
  };
}

const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({ 
  className = '', 
  filters = {} 
}) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<'pie' | 'bar'>('pie');
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');

  useEffect(() => {
    fetchAnalytics();
  }, [filters, timeRange]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));

      setStats(staticDashboardStats);
      
      // Use static analytics data
      const analytics = staticAnalyticsData;
      setChartData({
        categoryDistribution: {
          categories: Object.keys(analytics.categoryDistribution),
          counts: Object.values(analytics.categoryDistribution)
        },
        priorityDistribution: {
          priorities: Object.keys(analytics.priorityDistribution),
          counts: Object.values(analytics.priorityDistribution)
        },
        timeSeriesData: {
          dates: analytics.timeSeriesData.map(item => item.date),
          counts: analytics.timeSeriesData.map(item => item.count)
        },
        senderAnalytics: {
          senders: analytics.topSenders.map(item => item.sender),
          counts: analytics.topSenders.map(item => item.count)
        }
      });
    } catch (err) {
      console.error('Error fetching analytics:', err);
      setError('Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  const exportChart = (chartTitle: string) => {
    // This would implement chart export functionality
    console.log(`Exporting ${chartTitle} chart...`);
  };

  const getCategoryColors = () => ({
    'Work': '#3B82F6',
    'Personal': '#10B981',
    'Finance': '#8B5CF6',
    'Shopping': '#EC4899',
    'Travel': '#6366F1',
    'Urgent': '#EF4444',
    'Other': '#6B7280'
  });

  const getPriorityColors = () => ({
    'High': '#EF4444',
    'Medium': '#F59E0B',
    'Low': '#10B981'
  });

  if (loading) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="flex items-center gap-2 text-gray-600">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span>Loading analytics...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`p-6 bg-red-50 border border-red-200 rounded-lg ${className}`}>
        <div className="flex items-center gap-2 text-red-800">
          <Mail className="w-5 h-5" />
          <span>{error}</span>
        </div>
        <button
          onClick={fetchAnalytics}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Email Analytics</h2>
          <p className="text-gray-600">Insights and trends from your email data</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="1y">Last year</option>
          </select>
          <button
            onClick={fetchAnalytics}
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Emails</p>
                <p className="text-2xl font-bold text-gray-900">{stats.totalEmails.toLocaleString()}</p>
              </div>
              <Mail className="w-8 h-8 text-blue-600" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Unread Emails</p>
                <p className="text-2xl font-bold text-gray-900">{stats.unreadEmails.toLocaleString()}</p>
                <p className="text-sm text-gray-500">
                  {stats.totalEmails > 0 
                    ? `${((stats.unreadEmails / stats.totalEmails) * 100).toFixed(1)}% unread`
                    : '0% unread'
                  }
                </p>
              </div>
              <Mail className="w-8 h-8 text-orange-600" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Categories</p>
                <p className="text-2xl font-bold text-gray-900">{stats.categories.length}</p>
              </div>
              <Filter className="w-8 h-8 text-green-600" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg per Day</p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats.averagePerDay ? stats.averagePerDay.toFixed(1) : '0'}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-600" />
            </div>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      {chartData && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Category Distribution */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Category Distribution</h3>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setChartType(chartType === 'pie' ? 'bar' : 'pie')}
                    className="p-2 hover:bg-gray-100 rounded"
                    title="Toggle chart type"
                  >
                    {chartType === 'pie' ? <BarChart3 className="w-4 h-4" /> : <PieChart className="w-4 h-4" />}
                  </button>
                  <button
                    onClick={() => exportChart('Category Distribution')}
                    className="p-2 hover:bg-gray-100 rounded"
                    title="Export chart"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {chartType === 'pie' ? (
                <Plot
                  data={[
                    {
                      values: chartData.categoryDistribution.counts,
                      labels: chartData.categoryDistribution.categories,
                      type: 'pie',
                      textinfo: 'label+percent',
                      textposition: 'outside',
                      automargin: true,
                      marker: {
                        colors: chartData.categoryDistribution.categories.map(
                          cat => getCategoryColors()[cat as keyof typeof getCategoryColors] || '#6B7280'
                        )
                      }
                    }
                  ]}
                  layout={{
                    height: 300,
                    margin: { t: 20, b: 20, l: 20, r: 20 },
                    showlegend: false,
                    font: { size: 12 }
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              ) : (
                <Plot
                  data={[
                    {
                      x: chartData.categoryDistribution.categories,
                      y: chartData.categoryDistribution.counts,
                      type: 'bar',
                      marker: {
                        color: chartData.categoryDistribution.categories.map(
                          cat => getCategoryColors()[cat as keyof typeof getCategoryColors] || '#6B7280'
                        )
                      }
                    }
                  ]}
                  layout={{
                    height: 300,
                    margin: { t: 20, b: 60, l: 40, r: 20 },
                    xaxis: { title: 'Category' },
                    yaxis: { title: 'Count' },
                    font: { size: 12 }
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              )}
            </div>

            {/* Priority Distribution */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-red-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Priority Distribution</h3>
                </div>
                <button
                  onClick={() => exportChart('Priority Distribution')}
                  className="p-2 hover:bg-gray-100 rounded"
                  title="Export chart"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>

              <Plot
                data={[
                  {
                    values: chartData.priorityDistribution.counts,
                    labels: chartData.priorityDistribution.priorities,
                    type: 'pie',
                    textinfo: 'label+percent',
                    textposition: 'outside',
                    automargin: true,
                    marker: {
                      colors: chartData.priorityDistribution.priorities.map(
                        priority => getPriorityColors()[priority as keyof typeof getPriorityColors] || '#6B7280'
                      )
                    }
                  }
                ]}
                layout={{
                  height: 300,
                  margin: { t: 20, b: 20, l: 20, r: 20 },
                  showlegend: false,
                  font: { size: 12 }
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Time Series */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-green-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Email Volume Over Time</h3>
                </div>
                <button
                  onClick={() => exportChart('Email Volume Over Time')}
                  className="p-2 hover:bg-gray-100 rounded"
                  title="Export chart"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>

              <Plot
                data={[
                  {
                    x: chartData.timeSeriesData.dates,
                    y: chartData.timeSeriesData.counts,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#3B82F6', width: 2 },
                    marker: { color: '#3B82F6', size: 6 },
                    fill: 'tonexty',
                    fillcolor: 'rgba(59, 130, 246, 0.1)'
                  }
                ]}
                layout={{
                  height: 300,
                  margin: { t: 20, b: 60, l: 40, r: 20 },
                  xaxis: { 
                    title: 'Date',
                    type: 'date'
                  },
                  yaxis: { title: 'Email Count' },
                  font: { size: 12 },
                  hovermode: 'x unified'
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Top Senders */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Users className="w-5 h-5 text-purple-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Top Senders</h3>
                </div>
                <button
                  onClick={() => exportChart('Top Senders')}
                  className="p-2 hover:bg-gray-100 rounded"
                  title="Export chart"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>

              <Plot
                data={[
                  {
                    y: chartData.senderAnalytics.senders.slice(0, 10),
                    x: chartData.senderAnalytics.counts.slice(0, 10),
                    type: 'bar',
                    orientation: 'h',
                    marker: { color: '#8B5CF6' }
                  }
                ]}
                layout={{
                  height: 300,
                  margin: { t: 20, b: 40, l: 120, r: 20 },
                  xaxis: { title: 'Email Count' },
                  yaxis: { 
                    title: 'Sender',
                    automargin: true
                  },
                  font: { size: 12 }
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Hourly Distribution Chart */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-indigo-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Email Activity by Hour</h3>
                </div>
                <button
                  onClick={() => exportChart('Hourly Distribution')}
                  className="p-2 hover:bg-gray-100 rounded"
                  title="Export chart"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>

              <Plot
                data={[
                  {
                    x: staticAnalyticsData.hourlyDistribution.map(item => `${item.hour}:00`),
                    y: staticAnalyticsData.hourlyDistribution.map(item => item.count),
                    type: 'bar',
                    marker: { color: '#6366F1' }
                  }
                ]}
                layout={{
                  height: 300,
                  margin: { t: 20, b: 60, l: 40, r: 20 },
                  xaxis: { title: 'Hour of Day' },
                  yaxis: { title: 'Email Count' },
                  font: { size: 12 }
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Weekly Distribution Chart */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-teal-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Weekly Distribution</h3>
                </div>
                <button
                  onClick={() => exportChart('Weekly Distribution')}
                  className="p-2 hover:bg-gray-100 rounded"
                  title="Export chart"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>

              <Plot
                data={[
                  {
                    x: staticAnalyticsData.weeklyDistribution.map(item => item.day),
                    y: staticAnalyticsData.weeklyDistribution.map(item => item.count),
                    type: 'bar',
                    marker: { color: '#14B8A6' }
                  }
                ]}
                layout={{
                  height: 300,
                  margin: { t: 20, b: 60, l: 40, r: 20 },
                  xaxis: { title: 'Day of Week' },
                  yaxis: { title: 'Email Count' },
                  font: { size: 12 }
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </>
      )}

      {/* Additional Insights */}
      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Insights</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-blue-800">Most Active Category</p>
                <p className="text-lg font-bold text-blue-900">Work</p>
                <p className="text-xs text-blue-600">423 emails (33.9%)</p>
              </div>
              
              <div className="p-4 bg-green-50 rounded-lg">
                <p className="text-sm font-medium text-green-800">Read Rate</p>
                <p className="text-lg font-bold text-green-900">
                  {stats.totalEmails > 0 
                    ? `${(((stats.totalEmails - stats.unreadEmails) / stats.totalEmails) * 100).toFixed(1)}%`
                    : '0%'
                  }
                </p>
                <p className="text-xs text-green-600">Above average</p>
              </div>

              <div className="p-4 bg-purple-50 rounded-lg">
                <p className="text-sm font-medium text-purple-800">Avg Response Time</p>
                <p className="text-lg font-bold text-purple-900">
                  {stats.averageResponseTime || '2.4 hours'}
                </p>
                <p className="text-xs text-purple-600">Improved by 15%</p>
              </div>

              <div className="p-4 bg-orange-50 rounded-lg">
                <p className="text-sm font-medium text-orange-800">Peak Hour</p>
                <p className="text-lg font-bold text-orange-900">2:00 PM</p>
                <p className="text-xs text-orange-600">55 emails received</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Analysis</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Positive</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '36.6%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">456 (36.6%)</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Neutral</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-gray-500 h-2 rounded-full" style={{ width: '49.9%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">623 (49.9%)</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Negative</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-red-500 h-2 rounded-full" style={{ width: '13.5%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">168 (13.5%)</span>
                </div>
              </div>
            </div>
            
            <div className="mt-6 pt-4 border-t border-gray-200">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Response Time by Category</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-red-600">Urgent</span>
                  <span className="font-medium">0.5 hours</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-600">Work</span>
                  <span className="font-medium">1.2 hours</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-purple-600">Finance</span>
                  <span className="font-medium">2.1 hours</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-indigo-600">Travel</span>
                  <span className="font-medium">3.4 hours</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyticsCharts;