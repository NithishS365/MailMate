// Email-related types
export interface Email {
  id: string;
  from_address: string;
  to_address: string;
  cc_address?: string;
  bcc_address?: string;
  subject: string;
  body: string;
  timestamp: string;
  category: string;
  priority: string;
  attachments: string[];
  is_read: boolean;
  folder: string;
}

export interface EmailFilters {
  category?: string;
  priority?: string;
  dateFrom?: string;
  dateTo?: string;
  search?: string;
  folder?: string;
  isRead?: boolean;
}

export interface EmailStats {
  total: number;
  unread: number;
  categories: Record<string, number>;
  priorities: Record<string, number>;
  recentActivity: Array<{
    date: string;
    count: number;
  }>;
}

// Classification and analysis types
export interface ClassificationResult {
  predicted_category: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface SummaryResult {
  success: boolean;
  summary: string;
  original_length: number;
  summary_length: number;
  compression_ratio: number;
  processing_time: number;
  error_message?: string;
}

// Chat and AI types
export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  type?: 'text' | 'chart' | 'email_list';
  data?: any;
}

export interface AIQueryRequest {
  query: string;
  context?: {
    currentEmails?: Email[];
    filters?: EmailFilters;
  };
  includeEmailData?: boolean;
}

export interface AIQueryResponse {
  response: string;
  answer: string;
  type: 'text' | 'chart' | 'email_list' | 'action';
  data?: any;
  suggestions?: string[];
}

// Dashboard and UI types
export interface DashboardStats {
  totalEmails: number;
  unreadEmails: number;
  categories: string[];
  priorityDistribution: Record<string, number>;
  categoryDistribution: Record<string, number>;
  emailsPerDay: Array<{
    date: string;
    count: number;
  }>;
  avgProcessingTime: number;
  averagePerDay?: number;
  averageResponseTime?: string;
  lastUpdate: string;
}

export interface NotificationItem {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success' | 'email' | 'system';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

// WebSocket event types
export interface SocketEvents {
  'email_received': (email: Email) => void;
  'email_updated': (email: Email) => void;
  'stats_updated': (stats: DashboardStats) => void;
  'notification': (notification: NotificationItem) => void;
  'processing_complete': (data: any) => void;
}

// API response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

// Chart data types
export interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string[];
    borderColor?: string[];
    borderWidth?: number;
  }>;
}

export interface PlotlyChartData {
  x: any[];
  y: any[];
  type: string;
  name?: string;
  marker?: any;
  mode?: string;
  line?: any;
}

// Filter and search types
export interface SearchOptions {
  query: string;
  filters: EmailFilters;
  sortBy: 'timestamp' | 'subject' | 'priority' | 'category';
  sortOrder: 'asc' | 'desc';
  page: number;
  pageSize: number;
}

export interface AutocompleteOption {
  value: string;
  label: string;
  category?: string;
}

// Theme and UI state types
export interface ThemeConfig {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  success: string;
  warning: string;
  error: string;
  info: string;
}

export interface UIState {
  sidebarOpen: boolean;
  chatOpen: boolean;
  notificationsOpen: boolean;
  currentView: 'dashboard' | 'emails' | 'analytics' | 'settings';
  loading: boolean;
  error?: string;
}

// Configuration types
export interface AppConfig {
  apiBaseUrl: string;
  wsUrl: string;
  enableRealTimeUpdates: boolean;
  refreshInterval: number;
  pageSize: number;
  maxChatHistory: number;
  theme: 'light' | 'dark' | 'auto';
}

// Utility types
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Component prop types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingState {
  isLoading: boolean;
  error?: string;
  lastUpdated?: string;
}