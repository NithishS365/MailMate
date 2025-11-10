import React, { useState, useEffect } from 'react';
import {
  Mail,
  BarChart3,
  MessageCircle,
  Settings,
  Bell,
  Search,
  Filter,
  Grid,
  Menu,
  X
} from 'lucide-react';
import EmailBrowser from './EmailBrowser';
import AnalyticsCharts from './AnalyticsCharts';
import AIChat from './AIChat';
import { useDashboardStats, useNotifications } from '../hooks';
import { EmailFilters } from '../types';

interface DashboardProps {
  className?: string;
}

type ViewMode = 'overview' | 'emails' | 'analytics' | 'chat';

const Dashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  const [currentView, setCurrentView] = useState<ViewMode>('overview');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [filters] = useState<EmailFilters>({});
  const [isMobile, setIsMobile] = useState(false);

  const { stats, loading, fetchStats } = useDashboardStats();
  const { notifications, unreadCount } = useNotifications();

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    fetchStats();
  }, [filters, fetchStats]);

  const navigationItems = [
    {
      id: 'overview' as ViewMode,
      name: 'Overview',
      icon: Grid,
      badge: null
    },
    {
      id: 'emails' as ViewMode,
      name: 'Email Browser',
      icon: Mail,
      badge: stats?.unreadEmails || null
    },
    {
      id: 'analytics' as ViewMode,
      name: 'Analytics',
      icon: BarChart3,
      badge: null
    },
    {
      id: 'chat' as ViewMode,
      name: 'AI Assistant',
      icon: MessageCircle,
      badge: null
    }
  ];

  const handleViewChange = (view: ViewMode) => {
    setCurrentView(view);
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard Overview</h1>
          <p className="text-gray-600">Welcome to your email management center</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg">
            <Settings className="w-5 h-5" />
          </button>
          <button className="relative p-2 text-gray-600 hover:bg-gray-100 rounded-lg">
            <Bell className="w-5 h-5" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Emails</p>
                <p className="text-2xl font-bold text-gray-900">{stats.totalEmails.toLocaleString()}</p>
              </div>
              <Mail className="w-8 h-8 text-blue-600" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Unread</p>
                <p className="text-2xl font-bold text-orange-600">{stats.unreadEmails.toLocaleString()}</p>
                <p className="text-sm text-gray-500">
                  {stats.totalEmails > 0 
                    ? `${((stats.unreadEmails / stats.totalEmails) * 100).toFixed(1)}%`
                    : '0%'
                  }
                </p>
              </div>
              <Mail className="w-8 h-8 text-orange-600" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Categories</p>
                <p className="text-2xl font-bold text-green-600">{stats.categories.length}</p>
              </div>
              <Filter className="w-8 h-8 text-green-600" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Daily Average</p>
                <p className="text-2xl font-bold text-purple-600">
                  {stats.averagePerDay ? stats.averagePerDay.toFixed(1) : '0'}
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-600" />
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <button
              onClick={() => handleViewChange('emails')}
              className="w-full flex items-center gap-3 p-3 text-left bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors"
            >
              <Mail className="w-5 h-5 text-blue-600" />
              <div>
                <p className="font-medium text-blue-900">Browse Emails</p>
                <p className="text-sm text-blue-700">Search and manage your emails</p>
              </div>
            </button>
            <button
              onClick={() => handleViewChange('analytics')}
              className="w-full flex items-center gap-3 p-3 text-left bg-green-50 hover:bg-green-100 rounded-lg transition-colors"
            >
              <BarChart3 className="w-5 h-5 text-green-600" />
              <div>
                <p className="font-medium text-green-900">View Analytics</p>
                <p className="text-sm text-green-700">Analyze email patterns and trends</p>
              </div>
            </button>
            <button
              onClick={() => handleViewChange('chat')}
              className="w-full flex items-center gap-3 p-3 text-left bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors"
            >
              <MessageCircle className="w-5 h-5 text-purple-600" />
              <div>
                <p className="font-medium text-purple-900">AI Assistant</p>
                <p className="text-sm text-purple-700">Get help with email management</p>
              </div>
            </button>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            {notifications.slice(0, 5).map((notification) => (
              <div key={notification.id} className="flex items-start gap-3 p-2 hover:bg-gray-50 rounded">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  notification.type === 'email' ? 'bg-blue-500' :
                  notification.type === 'system' ? 'bg-green-500' : 'bg-orange-500'
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {notification.title}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(notification.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            {notifications.length === 0 && (
              <p className="text-sm text-gray-500">No recent activity</p>
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Email Insights</h3>
          <div className="space-y-4">
            {stats && (
              <>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Read Rate</span>
                    <span className="font-medium">
                      {stats.totalEmails > 0 
                        ? `${(((stats.totalEmails - stats.unreadEmails) / stats.totalEmails) * 100).toFixed(1)}%`
                        : '0%'
                      }
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ 
                        width: stats.totalEmails > 0 
                          ? `${((stats.totalEmails - stats.unreadEmails) / stats.totalEmails) * 100}%`
                          : '0%'
                      }}
                    />
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-600 mb-2">Top Categories</p>
                  <div className="space-y-1">
                    {stats.categories.slice(0, 3).map((category, index) => (
                      <div key={category} className="flex justify-between text-sm">
                        <span className="text-gray-700">{category}</span>
                        <span className="text-gray-500">#{index + 1}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Mini Analytics Preview */}
      <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Analytics Preview</h3>
          <button
            onClick={() => handleViewChange('analytics')}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
          >
            View Full Analytics →
          </button>
        </div>
        <AnalyticsCharts className="h-64" filters={filters} />
      </div>
    </div>
  );

  const renderContent = () => {
    switch (currentView) {
      case 'emails':
        return <EmailBrowser className="h-full" />;
      case 'analytics':
        return <AnalyticsCharts className="h-full" filters={filters} />;
      case 'chat':
        return <AIChat className="h-full" />;
      case 'overview':
      default:
        return renderOverview();
    }
  };

  return (
    <div className={`h-screen bg-gray-50 flex ${className}`}>
      {/* Sidebar */}
      <div className={`${
        sidebarOpen ? 'w-64' : 'w-0'
      } ${
        isMobile ? 'fixed inset-y-0 left-0 z-50' : 'relative'
      } bg-white border-r border-gray-200 transition-all duration-300 overflow-hidden`}>
        
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Mail className="w-8 h-8 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-900">MailMate</h2>
            </div>
            {isMobile && (
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-4">
          <ul className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentView === item.id;
              
              return (
                <li key={item.id}>
                  <button
                    onClick={() => handleViewChange(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{item.name}</span>
                    {item.badge && (
                      <span className="ml-auto bg-red-500 text-white text-xs rounded-full px-2 py-1">
                        {item.badge > 99 ? '99+' : item.badge}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Sidebar Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
          <div className="text-sm text-gray-500">
            <p>MailMate Dashboard</p>
            <p className="text-xs">v1.0.0</p>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="bg-white border-b border-gray-200 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <Menu className="w-5 h-5" />
              </button>
              <h1 className="text-lg font-semibold text-gray-900 capitalize">
                {currentView === 'overview' ? 'Dashboard' : currentView}
              </h1>
            </div>

            <div className="flex items-center gap-3">
              {currentView !== 'overview' && (
                <div className="flex items-center gap-2">
                  <Search className="w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search..."
                    className="px-3 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              )}
              
              <button className="relative p-2 text-gray-600 hover:bg-gray-100 rounded-lg">
                <Bell className="w-5 h-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                    {unreadCount > 9 ? '9' : unreadCount}
                  </span>
                )}
              </button>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-hidden">
          <div className="h-full p-6 overflow-y-auto">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;