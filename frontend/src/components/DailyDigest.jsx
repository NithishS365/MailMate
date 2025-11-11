import React, { useState, useEffect } from 'react';
import { digestService } from '../services/api';
import { FaVolumeUp } from 'react-icons/fa';

const DailyDigest = () => {
  const [digest, setDigest] = useState(null);
  const [loading, setLoading] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState(null);

  const loadDigest = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await digestService.getDailyDigest();
      setDigest(data);
    } catch (err) {
      setError('Failed to load daily digest');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const speakDigest = async () => {
    if (!digest || digest.emails.length === 0) {
      setError('No emails to read');
      return;
    }

    setSpeaking(true);
    setError(null);

    try {
      // Use Web Speech API for browser TTS
      if ('speechSynthesis' in window) {
        // Create summary text from digest emails
        let summaryText = `You have ${digest.emails.length} priority emails. `;
        
        digest.emails.forEach((email, index) => {
          summaryText += `Email ${index + 1}: ${email.subject} from ${email.sender}. `;
          // Add a short preview of the body
          const preview = email.body.substring(0, 100).replace(/[^\w\s]/gi, ' ');
          summaryText += `${preview}. `;
        });

        const utterance = new SpeechSynthesisUtterance(summaryText);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 1;
        utterance.lang = 'en-US';
        
        utterance.onend = () => {
          setSpeaking(false);
        };

        utterance.onerror = (event) => {
          console.error('Speech synthesis error:', event);
          setSpeaking(false);
          setError('Failed to speak. Please try again.');
        };
        
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        // Start speaking
        window.speechSynthesis.speak(utterance);
      } else {
        setError('Text-to-Speech is not supported in your browser');
        setSpeaking(false);
      }
    } catch (err) {
      setError('Failed to generate speech');
      console.error(err);
      setSpeaking(false);
    }
  };

  const stopSpeaking = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setSpeaking(false);
  };

  useEffect(() => {
    loadDigest();
  }, []);

  if (loading) {
    return <div className="loading">Loading daily digest...</div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <h2>📋 Daily Digest</h2>
        <div style={{ display: 'flex', gap: '10px' }}>
          {!speaking ? (
            <button
              className="btn btn-primary"
              onClick={speakDigest}
              disabled={!digest || digest.emails.length === 0}
              title="Read emails aloud using Text-to-Speech"
            >
              <FaVolumeUp /> Read Aloud
            </button>
          ) : (
            <button className="btn btn-secondary" onClick={stopSpeaking}>
              Stop Speaking
            </button>
          )}
        </div>
      </div>

      {error && <div className="notification error">{error}</div>}

      {digest && digest.emails && digest.emails.length > 0 ? (
        <div className="digest-section">
          <p style={{ color: '#666', marginBottom: '15px', fontSize: '0.95rem' }}>
            📌 Your top {digest.count} most important emails based on AI priority analysis:
          </p>
          {digest.emails.map((email, index) => (
            <div key={email.id} className="digest-email">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                <h4 style={{ margin: 0, flex: 1 }}>
                  {index + 1}. {email.subject}
                </h4>
                <span
                  className={`badge badge-${email.priority.toLowerCase()}`}
                  style={{ marginLeft: '10px' }}
                >
                  {email.priority}
                </span>
              </div>
              <p style={{ fontSize: '0.9rem', color: '#555', marginBottom: '4px' }}>
                <strong>From:</strong> {email.sender} ({email.sender_email})
              </p>
              <p style={{ fontSize: '0.85rem', color: '#666', marginBottom: '8px' }}>
                <strong>Priority Score:</strong> {(email.priority_score * 100).toFixed(0)}% | 
                <strong> Category:</strong> {email.category}
              </p>
              <p style={{ marginTop: '10px', fontSize: '0.9rem', color: '#444', lineHeight: '1.6' }}>
                {email.body.substring(0, 200)}
                {email.body.length > 200 ? '...' : ''}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state-icon">📭</div>
          <h3>No priority emails yet</h3>
          <p>Your daily digest will appear here once emails are available</p>
        </div>
      )}
    </div>
  );
};

export default DailyDigest;
