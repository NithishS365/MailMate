import { useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/api';
import wsService from '../services/websocket';
import { staticEmails, staticDashboardStats, staticNotifications, staticAnalyticsData } from '../services/staticData';
import { 
  Email, 
  EmailFilters, 
  DashboardStats, 
  SearchOptions, 
  LoadingState,
  NotificationItem 
} from '../types';

// Hook for managing email data with search and filtering
export const useEmails = (initialOptions: SearchOptions) => {
  const [emails, setEmails] = useState<Email[]>([]);
  const [loading, setLoading] = useState<LoadingState>({
    isLoading: false,
    error: undefined,
    lastUpdated: undefined,
  });
  const [pagination, setPagination] = useState({
    page: 1,
    pageSize: 20,
    total: 0,
    totalPages: 0,
  });

  const fetchEmails = useCallback(async (options: SearchOptions) => {
    setLoading(prev => ({ ...prev, isLoading: true, error: undefined }));
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    try {
      let filteredEmails = [...staticEmails];
      
      // Apply search filter
      if (options.query) {
        const query = options.query.toLowerCase();
        filteredEmails = filteredEmails.filter(email => 
          email.subject.toLowerCase().includes(query) ||
          email.body.toLowerCase().includes(query) ||
          email.from_address.toLowerCase().includes(query)
        );
      }
      
      // Apply category filter
      if (options.filters.category) {
        filteredEmails = filteredEmails.filter(email => 
          email.category === options.filters.category
        );
      }
      
      // Apply priority filter
      if (options.filters.priority) {
        filteredEmails = filteredEmails.filter(email => 
          email.priority === options.filters.priority
        );
      }
      
      // Apply read status filter
      if (options.filters.isRead !== undefined) {
        filteredEmails = filteredEmails.filter(email => 
          email.is_read === options.filters.isRead
        );
      }
      
      // Apply folder filter
      if (options.filters.folder) {
        filteredEmails = filteredEmails.filter(email => 
          email.folder === options.filters.folder
        );
      }
      
      // Apply sorting
      filteredEmails.sort((a, b) => {
        let aValue, bValue;
        switch (options.sortBy) {
          case 'timestamp':
            aValue = new Date(a.timestamp).getTime();
            bValue = new Date(b.timestamp).getTime();
            break;
          case 'subject':
            aValue = a.subject.toLowerCase();
            bValue = b.subject.toLowerCase();
            break;
          case 'priority':
            const priorityOrder = { 'High': 3, 'Medium': 2, 'Low': 1 };
            aValue = priorityOrder[a.priority as keyof typeof priorityOrder] || 0;
            bValue = priorityOrder[b.priority as keyof typeof priorityOrder] || 0;
            break;
          case 'category':
            aValue = a.category.toLowerCase();
            bValue = b.category.toLowerCase();
            break;
          default:
            aValue = new Date(a.timestamp).getTime();
            bValue = new Date(b.timestamp).getTime();
        }
        
        if (options.sortOrder === 'asc') {
          return aValue > bValue ? 1 : -1;
        } else {
          return aValue < bValue ? 1 : -1;
        }
      });
      
      // Apply pagination
      const total = filteredEmails.length;
      const totalPages = Math.ceil(total / options.pageSize);
      const startIndex = (options.page - 1) * options.pageSize;
      const endIndex = startIndex + options.pageSize;
      const paginatedEmails = filteredEmails.slice(startIndex, endIndex);
      
      setEmails(paginatedEmails);
      setPagination({
        page: options.page,
        pageSize: options.pageSize,
        total,
        totalPages
      });
      setLoading({
        isLoading: false,
        error: undefined,
        lastUpdated: new Date().toISOString(),
      });
    } catch (error) {
      setLoading({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        lastUpdated: new Date().toISOString(),
      });
    }
  }, []);

  const updateEmail = useCallback((updatedEmail: Email) => {
    setEmails(prev => 
      prev.map(email => 
        email.id === updatedEmail.id ? updatedEmail : email
      )
    );
  }, []);

  const addEmail = useCallback((newEmail: Email) => {
    setEmails(prev => [newEmail, ...prev]);
    setPagination(prev => ({ ...prev, total: prev.total + 1 }));
  }, []);

  const removeEmail = useCallback((emailId: string) => {
    setEmails(prev => prev.filter(email => email.id !== emailId));
    setPagination(prev => ({ ...prev, total: prev.total - 1 }));
  }, []);

  const markAsRead = useCallback(async (emailId: string) => {
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 200));
      const email = emails.find(e => e.id === emailId);
      if (email) {
        updateEmail({ ...email, is_read: true });
      }
    } catch (error) {
      console.error('Failed to mark email as read:', error);
    }
  }, [emails, updateEmail]);

  const deleteEmail = useCallback(async (emailId: string) => {
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 200));
      removeEmail(emailId);
    } catch (error) {
      console.error('Failed to delete email:', error);
    }
  }, [removeEmail]);

  return {
    emails,
    loading,
    pagination,
    fetchEmails,
    updateEmail,
    addEmail,
    removeEmail,
    markAsRead,
    deleteEmail,
  };
};

// Hook for real-time dashboard statistics
export const useDashboardStats = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();

  const fetchStats = useCallback(async (filters: EmailFilters = {}) => {
    try {
      setLoading(true);
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 300));
      setStats(staticDashboardStats);
      setError(undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();

    // Subscribe to real-time updates
    const handleStatsUpdate = (updatedStats: DashboardStats) => {
      setStats(updatedStats);
    };

    wsService.on('stats_updated', handleStatsUpdate);

    return () => {
      wsService.off('stats_updated', handleStatsUpdate);
    };
  }, [fetchStats]);

  return { stats, loading, error, refetch: fetchStats, fetchStats };
};

// Hook for WebSocket connection management
export const useWebSocket = () => {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    const handleConnectionStatus = (status: { connected: boolean; reason?: string }) => {
      setConnected(status.connected);
      if (!status.connected && status.reason) {
        setError(`Connection lost: ${status.reason}`);
      } else if (status.connected) {
        setError(undefined);
      }
    };

    const handleConnectionError = (errorData: { error: string }) => {
      setError(errorData.error);
    };

    wsService.on('connection_status', handleConnectionStatus);
    wsService.on('connection_error', handleConnectionError);

    // Set initial state
    setConnected(wsService.getConnectionStatus());

    return () => {
      wsService.off('connection_status', handleConnectionStatus);
      wsService.off('connection_error', handleConnectionError);
    };
  }, []);

  const reconnect = useCallback(() => {
    wsService.reconnect();
  }, []);

  return { connected, error, reconnect };
};

// Hook for notifications management
export const useNotifications = () => {
  const [notifications, setNotifications] = useState<NotificationItem[]>(staticNotifications);
  const [unreadCount, setUnreadCount] = useState(staticNotifications.filter(n => !n.read).length);

  const addNotification = useCallback((notification: NotificationItem) => {
    setNotifications(prev => [notification, ...prev]);
    if (!notification.read) {
      setUnreadCount(prev => prev + 1);
    }
  }, []);

  const markAsRead = useCallback((id: string) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === id 
          ? { ...notif, read: true }
          : notif
      )
    );
    setUnreadCount(prev => Math.max(0, prev - 1));
  }, []);

  const markAllAsRead = useCallback(() => {
    setNotifications(prev => 
      prev.map(notif => ({ ...notif, read: true }))
    );
    setUnreadCount(0);
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => {
      const notification = prev.find(n => n.id === id);
      if (notification && !notification.read) {
        setUnreadCount(count => Math.max(0, count - 1));
      }
      return prev.filter(n => n.id !== id);
    });
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
    setUnreadCount(0);
  }, []);

  useEffect(() => {
    // Listen for new notifications
    const handleNotification = (notification: NotificationItem) => {
      addNotification(notification);
    };

    const handleUrgentEmail = (email: Email) => {
      const urgentNotification: NotificationItem = {
        id: `urgent-${email.id}`,
        type: 'warning',
        title: 'Urgent Email Received',
        message: `From: ${email.from_address} - ${email.subject}`,
        timestamp: new Date().toISOString(),
        read: false,
        action: {
          label: 'View Email',
          onClick: () => {
            // Navigate to email or open modal
            console.log('Navigate to email:', email.id);
          },
        },
      };
      addNotification(urgentNotification);
    };

    wsService.on('notification', handleNotification);
    wsService.on('urgent_email', handleUrgentEmail);

    return () => {
      wsService.off('notification', handleNotification);
      wsService.off('urgent_email', handleUrgentEmail);
    };
  }, [addNotification]);

  return {
    notifications,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAll,
  };
};

// Hook for debounced search
export const useDebounce = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

// Hook for local storage state management
export const useLocalStorage = <T>(key: string, initialValue: T) => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue] as const;
};

// Hook for API loading states
export const useApiCall = <T>(
  apiCall: () => Promise<T>,
  dependencies: any[] = []
) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();

  const execute = useCallback(async () => {
    try {
      setLoading(true);
      setError(undefined);
      const result = await apiCall();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [apiCall, ...dependencies]);

  useEffect(() => {
    execute();
  }, [execute]);

  const refetch = useCallback(() => {
    execute();
  }, [execute]);

  return { data, loading, error, refetch };
};

// Hook for managing filters
export const useFilters = (initialFilters: EmailFilters) => {
  const [filters, setFilters] = useState<EmailFilters>(initialFilters);
  const [activeFilterCount, setActiveFilterCount] = useState(0);

  const updateFilter = useCallback((key: keyof EmailFilters, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value,
    }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(initialFilters);
  }, [initialFilters]);

  const clearFilter = useCallback((key: keyof EmailFilters) => {
    setFilters(prev => {
      const newFilters = { ...prev };
      delete newFilters[key];
      return newFilters;
    });
  }, []);

  useEffect(() => {
    const count = Object.values(filters).filter(
      value => value !== undefined && value !== '' && value !== null
    ).length;
    setActiveFilterCount(count);
  }, [filters]);

  return {
    filters,
    activeFilterCount,
    updateFilter,
    resetFilters,
    clearFilter,
  };
};

// Hook for polling data at intervals
export const usePolling = (
  callback: () => void,
  interval: number,
  enabled: boolean = true
) => {
  const callbackRef = useRef(callback);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (enabled && interval > 0) {
      intervalRef.current = setInterval(() => {
        callbackRef.current();
      }, interval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [interval, enabled]);

  const start = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(() => {
      callbackRef.current();
    }, interval);
  }, [interval]);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = undefined;
    }
  }, []);

  return { start, stop };
};

// Hook for AI Chat functionality
export const useAIChat = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();

  const sendQuery = useCallback(async (query: { query: string; context?: any; includeEmailData?: boolean }) => {
    try {
      setLoading(true);
      setError(undefined);
      
      const response = await apiService.sendChatQuery(query);
      return response.data;
    } catch (err) {
      console.error('Error sending AI query:', err);
      setError('Failed to process AI query');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    sendQuery,
    loading,
    error
  };
};