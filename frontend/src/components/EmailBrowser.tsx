import React, { useState, useEffect, useMemo } from 'react';
import { 
  Search, 
  Filter, 
  Mail, 
  MailOpen, 
  Clock, 
  User, 
  AlertCircle,
  Trash2,
  Eye,
  ChevronDown,
  ChevronRight,
  RefreshCw
} from 'lucide-react';
import { format } from 'date-fns';
import { Email, EmailFilters, SearchOptions } from '../types';
import { useEmails, useDebounce } from '../hooks';

interface EmailBrowserProps {
  className?: string;
}

const EmailBrowser: React.FC<EmailBrowserProps> = ({ className = '' }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<EmailFilters>({});
  const [showFilters, setShowFilters] = useState(false);
  const [selectedEmail, setSelectedEmail] = useState<Email | null>(null);
  const [sortBy, setSortBy] = useState<'timestamp' | 'subject' | 'priority' | 'category'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  
  const debouncedSearchQuery = useDebounce(searchQuery, 300);
  
  const searchOptions: SearchOptions = useMemo(() => ({
    query: debouncedSearchQuery,
    filters,
    sortBy,
    sortOrder,
    page: currentPage,
    pageSize: 20
  }), [debouncedSearchQuery, filters, sortBy, sortOrder, currentPage]);

  const {
    emails,
    loading,
    pagination,
    fetchEmails,
    markAsRead,
    deleteEmail
  } = useEmails(searchOptions);

  useEffect(() => {
    fetchEmails(searchOptions);
  }, [searchOptions, fetchEmails]);

  const handleFilterChange = (key: keyof EmailFilters, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value || undefined
    }));
    setCurrentPage(1);
  };

  const handleEmailClick = (email: Email) => {
    setSelectedEmail(email);
    if (!email.is_read) {
      markAsRead(email.id);
    }
  };

  const handleMarkAsRead = (emailId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    markAsRead(emailId);
  };

  const handleDeleteEmail = (emailId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (window.confirm('Are you sure you want to delete this email?')) {
      deleteEmail(emailId);
      if (selectedEmail?.id === emailId) {
        setSelectedEmail(null);
      }
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      'work': 'text-blue-600 bg-blue-100',
      'personal': 'text-green-600 bg-green-100',
      'finance': 'text-purple-600 bg-purple-100',
      'shopping': 'text-pink-600 bg-pink-100',
      'travel': 'text-indigo-600 bg-indigo-100',
      'urgent': 'text-red-600 bg-red-100'
    };
    return colors[category.toLowerCase() as keyof typeof colors] || 'text-gray-600 bg-gray-100';
  };

  const clearAllFilters = () => {
    setFilters({});
    setSearchQuery('');
    setCurrentPage(1);
  };

  const activeFilterCount = Object.values(filters).filter(v => v !== undefined && v !== '').length;

  return (
    <div className={`h-full flex flex-col bg-white ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Email Browser</h2>
          <button
            onClick={() => fetchEmails(searchOptions)}
            disabled={loading.isLoading}
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading.isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search emails..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 px-3 py-2 border rounded-lg ${
              showFilters ? 'bg-blue-50 border-blue-300' : 'border-gray-300 hover:bg-gray-50'
            }`}
          >
            <Filter className="w-4 h-4" />
            Filters
            {activeFilterCount > 0 && (
              <span className="bg-blue-600 text-white text-xs rounded-full px-2 py-1">
                {activeFilterCount}
              </span>
            )}
            {showFilters ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>

          {/* Sort Controls */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="timestamp">Sort by Date</option>
            <option value="subject">Sort by Subject</option>
            <option value="priority">Sort by Priority</option>
            <option value="category">Sort by Category</option>
          </select>

          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>

          {activeFilterCount > 0 && (
            <button
              onClick={clearAllFilters}
              className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg"
            >
              Clear All
            </button>
          )}
        </div>

        {/* Advanced Filters */}
        {showFilters && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  value={filters.category || ''}
                  onChange={(e) => handleFilterChange('category', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Categories</option>
                  <option value="Work">Work</option>
                  <option value="Personal">Personal</option>
                  <option value="Finance">Finance</option>
                  <option value="Shopping">Shopping</option>
                  <option value="Travel">Travel</option>
                  <option value="Urgent">Urgent</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                <select
                  value={filters.priority || ''}
                  onChange={(e) => handleFilterChange('priority', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Priorities</option>
                  <option value="High">High</option>
                  <option value="Medium">Medium</option>
                  <option value="Low">Low</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Read Status</label>
                <select
                  value={filters.isRead !== undefined ? filters.isRead.toString() : ''}
                  onChange={(e) => handleFilterChange('isRead', e.target.value === '' ? undefined : e.target.value === 'true')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Emails</option>
                  <option value="false">Unread Only</option>
                  <option value="true">Read Only</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Folder</label>
                <select
                  value={filters.folder || ''}
                  onChange={(e) => handleFilterChange('folder', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Folders</option>
                  <option value="INBOX">Inbox</option>
                  <option value="Sent">Sent</option>
                  <option value="Drafts">Drafts</option>
                  <option value="Spam">Spam</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 flex min-h-0">
        {/* Email List */}
        <div className="w-1/2 border-r border-gray-200 flex flex-col">
          {/* Stats */}
          <div className="px-4 py-2 bg-gray-50 border-b border-gray-200 text-sm text-gray-600">
            {loading.isLoading ? (
              'Loading...'
            ) : (
              `${pagination.total} emails found • Page ${pagination.page} of ${pagination.totalPages}`
            )}
          </div>

          {/* Email List */}
          <div className="flex-1 overflow-y-auto">
            {loading.isLoading ? (
              <div className="flex items-center justify-center h-32">
                <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
              </div>
            ) : emails.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-32 text-gray-500">
                <Mail className="w-8 h-8 mb-2" />
                <p>No emails found</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {emails.map((email) => (
                  <div
                    key={email.id}
                    onClick={() => handleEmailClick(email)}
                    className={`p-4 hover:bg-gray-50 cursor-pointer transition-colors ${
                      selectedEmail?.id === email.id ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                    } ${!email.is_read ? 'bg-blue-25' : ''}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2 min-w-0 flex-1">
                        {email.is_read ? (
                          <MailOpen className="w-4 h-4 text-gray-400 flex-shrink-0" />
                        ) : (
                          <Mail className="w-4 h-4 text-blue-600 flex-shrink-0" />
                        )}
                        <span className={`truncate ${!email.is_read ? 'font-semibold' : ''}`}>
                          {email.from_address}
                        </span>
                      </div>
                      <div className="flex items-center gap-1 flex-shrink-0">
                        <button
                          onClick={(e) => handleMarkAsRead(email.id, e)}
                          className="p-1 hover:bg-gray-200 rounded"
                          title={email.is_read ? 'Mark as unread' : 'Mark as read'}
                        >
                          <Eye className="w-3 h-3 text-gray-400" />
                        </button>
                        <button
                          onClick={(e) => handleDeleteEmail(email.id, e)}
                          className="p-1 hover:bg-red-100 rounded"
                          title="Delete email"
                        >
                          <Trash2 className="w-3 h-3 text-red-400" />
                        </button>
                      </div>
                    </div>

                    <div className={`mb-2 ${!email.is_read ? 'font-semibold' : ''}`}>
                      <p className="truncate text-gray-900">{email.subject}</p>
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded-full text-xs ${getCategoryColor(email.category)}`}>
                          {email.category}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs ${getPriorityColor(email.priority)}`}>
                          {email.priority}
                        </span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-500">
                        <Clock className="w-3 h-3" />
                        <span>{format(new Date(email.timestamp), 'MMM d, HH:mm')}</span>
                      </div>
                    </div>

                    <div className="mt-2 text-sm text-gray-600 line-clamp-2">
                      {email.body}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Pagination */}
          {pagination.totalPages > 1 && (
            <div className="p-4 border-t border-gray-200 flex items-center justify-between">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <span className="text-sm text-gray-600">
                Page {currentPage} of {pagination.totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(Math.min(pagination.totalPages, currentPage + 1))}
                disabled={currentPage === pagination.totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          )}
        </div>

        {/* Email Detail */}
        <div className="w-1/2 flex flex-col">
          {selectedEmail ? (
            <>
              <div className="p-4 border-b border-gray-200 bg-white">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {selectedEmail.subject}
                    </h3>
                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <div className="flex items-center gap-1">
                        <User className="w-4 h-4" />
                        <span>From: {selectedEmail.from_address}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        <span>{format(new Date(selectedEmail.timestamp), 'PPP p')}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <span className={`px-3 py-1 rounded-full text-sm ${getCategoryColor(selectedEmail.category)}`}>
                      {selectedEmail.category}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-sm ${getPriorityColor(selectedEmail.priority)}`}>
                      {selectedEmail.priority}
                    </span>
                  </div>
                </div>

                {selectedEmail.cc_address && (
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-medium">CC:</span> {selectedEmail.cc_address}
                  </div>
                )}

                {selectedEmail.attachments && selectedEmail.attachments.length > 0 && (
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-medium">Attachments:</span> {selectedEmail.attachments.join(', ')}
                  </div>
                )}
              </div>

              <div className="flex-1 p-4 overflow-y-auto bg-white">
                <div className="prose max-w-none">
                  <p className="whitespace-pre-wrap text-gray-800 leading-relaxed">
                    {selectedEmail.body}
                  </p>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <Mail className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p className="text-lg mb-2">Select an email to view</p>
                <p className="text-sm">Choose an email from the list to see its content</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {loading.error && (
        <div className="p-4 bg-red-50 border-t border-red-200">
          <div className="flex items-center gap-2 text-red-800">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">{loading.error}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmailBrowser;