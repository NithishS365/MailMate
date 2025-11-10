import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Bot,
  User,
  Loader2,
  AlertCircle,
  Trash2,
  Copy,
  ThumbsUp,
  ThumbsDown,
  MessageSquare
} from 'lucide-react';
import { staticEmails, staticDashboardStats } from '../services/staticData';
import { AIQueryRequest } from '../types';

interface AIChatProps {
  className?: string;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
  error?: string;
}

const AIChat: React.FC<AIChatProps> = ({ className = '' }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Static AI responses based on common queries
  const getAIResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('unread') || lowerQuery.includes('new')) {
      const unreadCount = staticEmails.filter(e => !e.is_read).length;
      return `You have ${unreadCount} unread emails. Here are the most recent ones:\n\n• ${staticEmails.filter(e => !e.is_read).slice(0, 3).map(e => `${e.subject} (from ${e.from_address})`).join('\n• ')}`;
    }
    
    if (lowerQuery.includes('category') || lowerQuery.includes('categories')) {
      const categories = Object.entries(staticDashboardStats.categoryDistribution)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5);
      return `Your email categories breakdown:\n\n${categories.map(([cat, count]) => `• ${cat}: ${count} emails (${((count/staticDashboardStats.totalEmails)*100).toFixed(1)}%)`).join('\n')}`;
    }
    
    if (lowerQuery.includes('urgent') || lowerQuery.includes('important')) {
      const urgentEmails = staticEmails.filter(e => e.priority === 'High' || e.category === 'Urgent');
      return `You have ${urgentEmails.length} urgent/high priority emails:\n\n${urgentEmails.slice(0, 3).map(e => `• ${e.subject} (${e.priority} priority)`).join('\n')}`;
    }
    
    if (lowerQuery.includes('sender') || lowerQuery.includes('who sends')) {
      const senders = staticEmails.reduce((acc, email) => {
        acc[email.from_address] = (acc[email.from_address] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      const topSenders = Object.entries(senders)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5);
      return `Your top email senders:\n\n${topSenders.map(([sender, count]) => `• ${sender}: ${count} emails`).join('\n')}`;
    }
    
    if (lowerQuery.includes('today') || lowerQuery.includes('recent')) {
      const recentEmails = staticEmails.slice(0, 5);
      return `Here are your most recent emails:\n\n${recentEmails.map(e => `• ${e.subject} (from ${e.from_address}) - ${e.category}`).join('\n')}`;
    }
    
    if (lowerQuery.includes('summary') || lowerQuery.includes('overview')) {
      return `Email Summary:\n\n• Total emails: ${staticDashboardStats.totalEmails}\n• Unread: ${staticDashboardStats.unreadEmails}\n• Most active category: Work (${staticDashboardStats.categoryDistribution.Work} emails)\n• Average per day: ${staticDashboardStats.averagePerDay}\n• Response time: ${staticDashboardStats.averageResponseTime}`;
    }
    
    if (lowerQuery.includes('work') || lowerQuery.includes('business')) {
      const workEmails = staticEmails.filter(e => e.category === 'Work');
      return `You have ${workEmails.length} work-related emails. Recent ones:\n\n${workEmails.slice(0, 3).map(e => `• ${e.subject} (${e.priority} priority)`).join('\n')}`;
    }
    
    if (lowerQuery.includes('finance') || lowerQuery.includes('money') || lowerQuery.includes('bank')) {
      const financeEmails = staticEmails.filter(e => e.category === 'Finance');
      return `You have ${financeEmails.length} finance-related emails:\n\n${financeEmails.slice(0, 3).map(e => `• ${e.subject} (from ${e.from_address})`).join('\n')}`;
    }
    
    // Default response
    return `I can help you with:\n\n• Finding specific emails\n• Analyzing email patterns\n• Categorizing messages\n• Checking unread emails\n• Summarizing conversations\n• Identifying urgent items\n\nTry asking about your unread emails, categories, or recent messages!`;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Add welcome message on component mount
    if (messages.length === 0) {
      setMessages([
        {
          id: 'welcome',
          type: 'assistant',
          content: "Hello! I'm your AI email assistant. I can help you:\n\n• Analyze your emails\n• Find specific messages\n• Summarize conversations\n• Categorize and prioritize emails\n• Answer questions about your email data\n\nWhat would you like to know about your emails?",
          timestamp: new Date()
        }
      ]);
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Simulate AI processing delay
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
      
      const aiResponse = getAIResponse(input.trim());
      
      // Remove loading message and add actual response
      setMessages(prev => {
        const filtered = prev.filter(msg => msg.id !== loadingMessage.id);
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 2).toString(),
          type: 'assistant',
          content: aiResponse,
          timestamp: new Date()
        };
        return [...filtered, assistantMessage];
      });

    } catch (err) {
      // Remove loading message and add error message
      setMessages(prev => {
        const filtered = prev.filter(msg => msg.id !== loadingMessage.id);
        const errorMessage: ChatMessage = {
          id: (Date.now() + 2).toString(),
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          error: 'Sorry, I encountered an error processing your request. Please try again.'
        };
        return [...filtered, errorMessage];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: 'welcome',
        type: 'assistant',
        content: "Chat cleared! How can I help you with your emails?",
        timestamp: new Date()
      }
    ]);
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    // You could add a toast notification here
  };

  const provideFeedback = (messageId: string, isPositive: boolean) => {
    // This would send feedback to your AI service
    console.log(`Feedback for message ${messageId}: ${isPositive ? 'positive' : 'negative'}`);
  };

  const formatMessage = (content: string) => {
    // Simple formatting for better readability
    return content
      .split('\n')
      .map((line, index) => (
        <div key={index} className={line.startsWith('•') ? 'ml-4' : ''}>
          {line}
        </div>
      ));
  };

  const suggestedQueries = [
    "Show me all unread emails from this week",
    "What are my most common email categories?",
    "Find emails about project deadlines",
    "Summarize my urgent emails",
    "Who sends me the most emails?",
    "What emails need immediate attention?"
  ];

  return (
    <div className={`h-full flex flex-col bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="w-6 h-6 text-blue-600" />
            <div>
              <h3 className="text-lg font-semibold text-gray-900">AI Email Assistant</h3>
              <p className="text-sm text-gray-600">Ask me anything about your emails</p>
            </div>
          </div>
          <button
            onClick={clearChat}
            className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
            title="Clear chat"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.type === 'assistant' && (
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                  <Bot className="w-5 h-5 text-blue-600" />
                </div>
              </div>
            )}

            <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : ''}`}>
              <div
                className={`p-3 rounded-lg ${
                  message.type === 'user'
                    ? 'bg-blue-600 text-white'
                    : message.error
                    ? 'bg-red-50 border border-red-200'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                {message.isLoading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Thinking...</span>
                  </div>
                ) : message.error ? (
                  <div className="flex items-center gap-2 text-red-700">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm">{message.error}</span>
                  </div>
                ) : (
                  <div className="text-sm whitespace-pre-wrap">
                    {formatMessage(message.content)}
                  </div>
                )}
              </div>

              {/* Message Actions */}
              {!message.isLoading && !message.error && message.type === 'assistant' && (
                <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                  <button
                    onClick={() => copyMessage(message.content)}
                    className="flex items-center gap-1 hover:text-gray-700"
                    title="Copy message"
                  >
                    <Copy className="w-3 h-3" />
                    Copy
                  </button>
                  <button
                    onClick={() => provideFeedback(message.id, true)}
                    className="flex items-center gap-1 hover:text-green-600"
                    title="Good response"
                  >
                    <ThumbsUp className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => provideFeedback(message.id, false)}
                    className="flex items-center gap-1 hover:text-red-600"
                    title="Poor response"
                  >
                    <ThumbsDown className="w-3 h-3" />
                  </button>
                  <span className="text-xs">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>

            {message.type === 'user' && (
              <div className="flex-shrink-0 order-3">
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-gray-600" />
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Suggested Queries */}
        {messages.length === 1 && (
          <div className="space-y-3">
            <p className="text-sm text-gray-600 font-medium">Try asking:</p>
            <div className="grid grid-cols-1 gap-2">
              {suggestedQueries.map((query, index) => (
                <button
                  key={index}
                  onClick={() => setInput(query)}
                  className="p-3 text-left text-sm bg-gray-50 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <MessageSquare className="w-4 h-4 inline mr-2 text-gray-400" />
                  {query}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me about your emails..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={1}
              style={{
                minHeight: '40px',
                maxHeight: '120px',
                resize: 'none'
              }}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>

        <div className="mt-2 text-xs text-gray-500">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>


    </div>
  );
};

export default AIChat;