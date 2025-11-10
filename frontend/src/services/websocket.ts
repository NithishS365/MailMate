import { io, Socket } from 'socket.io-client';
import { Email, DashboardStats, NotificationItem, SocketEvents } from '../types';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnected = false;
  private listeners: Map<string, Set<Function>> = new Map();

  constructor(private url: string = 'http://localhost:5000') {
    this.connect();
  }

  connect(): void {
    try {
      this.socket = io(this.url, {
        autoConnect: true,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
        timeout: 10000,
        transports: ['polling', 'websocket'],
        withCredentials: true,
        extraHeaders: {
          'Access-Control-Allow-Origin': 'http://localhost:3000'
        }
      });

      this.setupEventListeners();
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnected = false;
      this.emit('connection_status', { connected: false, reason });
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        this.emit('connection_error', { error: 'Failed to connect after multiple attempts' });
      }
    });

    // Email-related events
    this.socket.on('email_received', (email: Email) => {
      console.log('New email received:', email);
      this.emit('email_received', email);
    });

    this.socket.on('email_updated', (email: Email) => {
      console.log('Email updated:', email);
      this.emit('email_updated', email);
    });

    this.socket.on('email_deleted', (emailId: string) => {
      console.log('Email deleted:', emailId);
      this.emit('email_deleted', emailId);
    });

    // Stats and dashboard events
    this.socket.on('stats_updated', (stats: DashboardStats) => {
      console.log('Dashboard stats updated');
      this.emit('stats_updated', stats);
    });

    this.socket.on('classification_complete', (data: any) => {
      console.log('Email classification complete:', data);
      this.emit('classification_complete', data);
    });

    this.socket.on('summarization_complete', (data: any) => {
      console.log('Email summarization complete:', data);
      this.emit('summarization_complete', data);
    });

    // Notification events
    this.socket.on('notification', (notification: NotificationItem) => {
      console.log('New notification:', notification);
      this.emit('notification', notification);
    });

    this.socket.on('urgent_email', (email: Email) => {
      console.log('Urgent email received:', email);
      this.emit('urgent_email', email);
    });

    // Processing events
    this.socket.on('processing_start', (data: any) => {
      console.log('Processing started:', data);
      this.emit('processing_start', data);
    });

    this.socket.on('processing_progress', (data: any) => {
      console.log('Processing progress:', data);
      this.emit('processing_progress', data);
    });

    this.socket.on('processing_complete', (data: any) => {
      console.log('Processing complete:', data);
      this.emit('processing_complete', data);
    });

    // Error events
    this.socket.on('error', (error: any) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    });
  }

  // Event emission and listening
  on<K extends keyof SocketEvents>(event: K, callback: SocketEvents[K]): void;
  on(event: string, callback: Function): void;
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off<K extends keyof SocketEvents>(event: K, callback: SocketEvents[K]): void;
  off(event: string, callback: Function): void;
  off(event: string, callback: Function): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(callback);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  private emit(event: string, data?: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  // Send events to server
  send(event: string, data?: any): void {
    if (this.socket && this.isConnected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected, cannot send event:', event);
    }
  }

  // Join/leave rooms for targeted updates
  joinRoom(room: string): void {
    this.send('join_room', { room });
  }

  leaveRoom(room: string): void {
    this.send('leave_room', { room });
  }

  // Subscribe to email updates for specific filters
  subscribeToEmailUpdates(filters?: any): void {
    this.send('subscribe_emails', { filters });
  }

  // Subscribe to dashboard updates
  subscribeToDashboard(): void {
    this.send('subscribe_dashboard');
  }

  // Request real-time data refresh
  requestDataRefresh(): void {
    this.send('refresh_data');
  }

  // Connection management
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
    this.listeners.clear();
  }

  reconnect(): void {
    this.disconnect();
    this.reconnectAttempts = 0;
    setTimeout(() => this.connect(), 1000);
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  // Utility methods
  ping(): Promise<number> {
    return new Promise((resolve, reject) => {
      if (!this.socket || !this.isConnected) {
        reject(new Error('WebSocket not connected'));
        return;
      }

      const startTime = Date.now();
      
      this.socket.emit('ping', startTime);
      
      this.socket.once('pong', (timestamp: number) => {
        const latency = Date.now() - timestamp;
        resolve(latency);
      });

      // Timeout after 5 seconds
      setTimeout(() => {
        reject(new Error('Ping timeout'));
      }, 5000);
    });
  }

  // Debug methods
  getListeners(): Map<string, Set<Function>> {
    return this.listeners;
  }

  getSocketInfo(): any {
    if (!this.socket) return null;
    
    return {
      connected: this.isConnected,
      id: this.socket.id,
      reconnectAttempts: this.reconnectAttempts,
      url: this.url,
    };
  }
}

// Create and export singleton instance
const wsService = new WebSocketService(
  process.env.REACT_APP_WS_URL || 'http://localhost:5000'
);

export default wsService;