import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  Email, 
  EmailFilters, 
  EmailStats, 
  DashboardStats, 
  AIQueryRequest, 
  AIQueryResponse, 
  ApiResponse, 
  PaginatedResponse,
  SearchOptions,
  ClassificationResult,
  SummaryResult
} from '../types';

class ApiService {
  private api: AxiosInstance;
  private baseURL: string;
  private retryCount: number = 3;
  private retryDelay: number = 1000;

  constructor(baseURL: string = 'http://localhost:5000/api') {
    this.baseURL = baseURL;
    this.api = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      withCredentials: true,
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor with retry logic
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const config = error.config;
        
        // Only retry on network errors or 5xx server errors
        if (!config || !this.retryCount || 
            (error.response && error.response.status < 500)) {
          console.error('API Error:', error.response?.data || error.message);
          return Promise.reject(error);
        }

        config.__retryCount = config.__retryCount || 0;
        
        if (config.__retryCount >= this.retryCount) {
          console.error('API Error: Max retries reached', error.response?.data || error.message);
          return Promise.reject(error);
        }

        config.__retryCount += 1;
        console.log(`Retrying request (${config.__retryCount}/${this.retryCount})`);
        
        await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        return this.api(config);
      }
    );
  }

  // Email endpoints
  async getEmails(options: SearchOptions): Promise<PaginatedResponse<Email>> {
    const response: AxiosResponse<PaginatedResponse<Email>> = await this.api.get('/emails', {
      params: {
        page: options.page,
        per_page: options.pageSize,
        category: options.filters?.category,
        ...options.filters,
      },
    });
    return response.data;
  }

  async getEmailById(id: string): Promise<ApiResponse<Email>> {
    const response: AxiosResponse<ApiResponse<Email>> = await this.api.get(`/emails/${id}`);
    return response.data;
  }

  async markEmailAsRead(id: string): Promise<ApiResponse<void>> {
    const response: AxiosResponse<ApiResponse<void>> = await this.api.patch(`/emails/${id}/read`);
    return response.data;
  }

  async deleteEmail(id: string): Promise<ApiResponse<void>> {
    const response: AxiosResponse<ApiResponse<void>> = await this.api.delete(`/emails/${id}`);
    return response.data;
  }

  async updateEmailCategory(id: string, category: string): Promise<ApiResponse<Email>> {
    const response: AxiosResponse<ApiResponse<Email>> = await this.api.patch(`/emails/${id}/category`, {
      category,
    });
    return response.data;
  }

  async updateEmailPriority(id: string, priority: string): Promise<ApiResponse<Email>> {
    const response: AxiosResponse<ApiResponse<Email>> = await this.api.patch(`/emails/${id}/priority`, {
      priority,
    });
    return response.data;
  }

  // Classification and ML endpoints
  async classifyEmail(text: string): Promise<ApiResponse<ClassificationResult>> {
    const response: AxiosResponse<ApiResponse<ClassificationResult>> = await this.api.post('/classify', {
      text,
    });
    return response.data;
  }

  async summarizeEmail(text: string): Promise<ApiResponse<SummaryResult>> {
    const response: AxiosResponse<ApiResponse<SummaryResult>> = await this.api.post('/summarize', {
      text,
    });
    return response.data;
  }

  async generateSyntheticEmails(count: number): Promise<ApiResponse<Email[]>> {
    const response: AxiosResponse<ApiResponse<Email[]>> = await this.api.post('/emails/generate', {
      count,
    });
    return response.data;
  }

  // Statistics and analytics endpoints
  async getEmailStats(): Promise<ApiResponse<EmailStats>> {
    const response: AxiosResponse<ApiResponse<EmailStats>> = await this.api.get('/stats/emails');
    return response.data;
  }

  async getDashboardStats(): Promise<ApiResponse<DashboardStats>> {
    const response: AxiosResponse<ApiResponse<DashboardStats>> = await this.api.get('/stats/dashboard');
    return response.data;
  }

  async getCategoryDistribution(): Promise<ApiResponse<Record<string, number>>> {
    const response: AxiosResponse<ApiResponse<Record<string, number>>> = await this.api.get('/stats/categories');
    return response.data;
  }

  async getPriorityDistribution(): Promise<ApiResponse<Record<string, number>>> {
    const response: AxiosResponse<ApiResponse<Record<string, number>>> = await this.api.get('/stats/priorities');
    return response.data;
  }

  async getEmailTrends(days: number = 30): Promise<ApiResponse<Array<{ date: string; count: number }>>> {
    const response: AxiosResponse<ApiResponse<Array<{ date: string; count: number }>>> = await this.api.get('/stats/trends', {
      params: { days },
    });
    return response.data;
  }

  async getEmailAnalytics(options: { filters?: EmailFilters; timeRange?: string }): Promise<ApiResponse<any>> {
    const response: AxiosResponse<ApiResponse<any>> = await this.api.get('/analytics/emails', {
      params: {
        ...options.filters,
        timeRange: options.timeRange
      },
    });
    return response.data;
  }

  // AI Chat endpoints
  async sendChatQuery(request: AIQueryRequest): Promise<ApiResponse<AIQueryResponse>> {
    const response: AxiosResponse<ApiResponse<AIQueryResponse>> = await this.api.post('/chat/query', request);
    return response.data;
  }

  async getChatSuggestions(): Promise<ApiResponse<string[]>> {
    const response: AxiosResponse<ApiResponse<string[]>> = await this.api.get('/chat/suggestions');
    return response.data;
  }

  async getChatHistory(): Promise<ApiResponse<any[]>> {
    const response: AxiosResponse<ApiResponse<any[]>> = await this.api.get('/chat/history');
    return response.data;
  }

  // Search and filtering endpoints
  async searchEmails(query: string, filters?: EmailFilters): Promise<ApiResponse<Email[]>> {
    const response: AxiosResponse<ApiResponse<Email[]>> = await this.api.get('/emails/search', {
      params: {
        q: query,
        ...filters,
      },
    });
    return response.data;
  }

  async getSearchSuggestions(query: string): Promise<ApiResponse<string[]>> {
    const response: AxiosResponse<ApiResponse<string[]>> = await this.api.get('/emails/search/suggestions', {
      params: { q: query },
    });
    return response.data;
  }

  async getFilters(): Promise<ApiResponse<{
    categories: string[];
    priorities: string[];
    folders: string[];
  }>> {
    const response = await this.api.get('/emails/filters');
    return response.data;
  }

  // Settings and configuration endpoints
  async getSettings(): Promise<ApiResponse<any>> {
    const response: AxiosResponse<ApiResponse<any>> = await this.api.get('/settings');
    return response.data;
  }

  async updateSettings(settings: any): Promise<ApiResponse<any>> {
    const response: AxiosResponse<ApiResponse<any>> = await this.api.put('/settings', settings);
    return response.data;
  }

  // Health and status endpoints
  async getHealth(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    const response: AxiosResponse<ApiResponse<{ status: string; timestamp: string }>> = await this.api.get('/health');
    return response.data;
  }

  async getSystemStatus(): Promise<ApiResponse<{
    status: string;
    timestamp: string;
    version: string;
    components: any;
    extensible_interfaces: any;
    features: any;
  }>> {
    const response = await this.api.get('/status');
    return response.data;
  }

  // File upload endpoints
  async uploadEmails(file: File): Promise<ApiResponse<{ imported: number; errors: string[] }>> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.api.post('/emails/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Export endpoints
  async exportEmails(filters?: EmailFilters, format: 'csv' | 'json' = 'csv'): Promise<Blob> {
    const response = await this.api.get('/emails/export', {
      params: {
        format,
        ...filters,
      },
      responseType: 'blob',
    });
    return response.data;
  }
}

const apiService = new ApiService();
export default apiService;