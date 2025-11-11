import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const emailService = {
  // Generate initial emails
  generateEmails: async () => {
    const response = await api.post('/emails/generate');
    return response.data;
  },

  // Get all emails with optional filters
  getEmails: async (params = {}) => {
    const response = await api.get('/emails/', { params });
    return response.data;
  },

  // Get single email
  getEmail: async (id) => {
    const response = await api.get(`/emails/${id}`);
    return response.data;
  },

  // Update email
  updateEmail: async (id, data) => {
    const response = await api.put(`/emails/${id}`, data);
    return response.data;
  },

  // Delete email
  deleteEmail: async (id) => {
    const response = await api.delete(`/emails/${id}`);
    return response.data;
  },

  // Get statistics
  getStatistics: async () => {
    const response = await api.get('/emails/stats/summary');
    return response.data;
  },

  // Get top priority emails
  getTopPriority: async (limit = 5) => {
    const response = await api.get('/emails/priority/top', {
      params: { limit }
    });
    return response.data;
  },

  // Retrain classifier
  retrainClassifier: async () => {
    const response = await api.post('/emails/retrain');
    return response.data;
  },
};

export const digestService = {
  // Get daily digest
  getDailyDigest: async () => {
    const response = await api.get('/digest/daily');
    return response.data;
  },

  // Generate speech for text
  generateSpeech: async (text) => {
    const response = await api.post('/digest/speak', null, {
      params: { text }
    });
    return response.data;
  },

  // Speak daily digest
  speakDigest: async () => {
    const response = await api.get('/digest/speak-digest');
    return response.data;
  },
};

export default api;
