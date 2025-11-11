import React, { useState } from 'react';
import { format } from 'date-fns';
import { FaStar, FaRegStar, FaArchive, FaTrash, FaExclamationCircle } from 'react-icons/fa';

const EmailList = ({ emails, onEmailClick, onDelete, onToggleStar, onArchive }) => {
  const [starred, setStarred] = useState(new Set());
  const [archived, setArchived] = useState(new Set());

  if (!emails || emails.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📧</div>
        <h3>No emails found</h3>
        <p>Generate emails to get started</p>
      </div>
    );
  }

  const handleStarClick = (e, emailId) => {
    e.stopPropagation();
    const newStarred = new Set(starred);
    if (newStarred.has(emailId)) {
      newStarred.delete(emailId);
    } else {
      newStarred.add(emailId);
    }
    setStarred(newStarred);
    onToggleStar && onToggleStar(emailId);
  };

  const handleArchive = (e, emailId) => {
    e.stopPropagation();
    const newArchived = new Set(archived);
    newArchived.add(emailId);
    setArchived(newArchived);
    onArchive && onArchive(emailId);
  };

  const handleDelete = (e, emailId) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this email?')) {
      onDelete && onDelete(emailId);
    }
  };

  const getCategoryClass = (category) => {
    const classes = {
      work: 'badge-work',
      personal: 'badge-personal',
      urgent: 'badge-urgent',
      promotion: 'badge-promotion',
    };
    return classes[category] || 'badge-work';
  };

  const getPriorityClass = (priority) => {
    const classes = {
      high: 'badge-high',
      medium: 'badge-medium',
      low: 'badge-low',
    };
    return classes[priority] || 'badge-medium';
  };

  return (
    <div className="email-list">
      {emails.filter(email => !archived.has(email.id)).map((email) => (
        <div
          key={email.id}
          className={`email-item ${email.is_read === 0 ? 'unread' : ''}`}
          onClick={() => onEmailClick(email)}
        >
          <div className="email-header">
            <div className="email-left">
              <button
                className={`star-btn ${starred.has(email.id) ? 'starred' : ''}`}
                onClick={(e) => handleStarClick(e, email.id)}
                title={starred.has(email.id) ? 'Unstar' : 'Star'}
              >
                {starred.has(email.id) ? <FaStar /> : <FaRegStar />}
              </button>
              <div className="email-sender">{email.sender}</div>
            </div>
            <div className="email-actions">
              <button
                className="action-btn archive-btn"
                onClick={(e) => handleArchive(e, email.id)}
                title="Archive"
              >
                <FaArchive />
              </button>
              <button
                className="action-btn delete-btn"
                onClick={(e) => handleDelete(e, email.id)}
                title="Delete"
              >
                <FaTrash />
              </button>
              <div className="email-time">
                {format(new Date(email.timestamp), 'MMM dd, HH:mm')}
              </div>
            </div>
          </div>
          <div className="email-subject">
            {email.is_read === 0 && <span className="unread-dot"></span>}
            {email.subject}
          </div>
          <div className="email-preview">{email.body}</div>
          <div className="email-meta">
            {email.category && (
              <span className={`badge ${getCategoryClass(email.category)}`}>
                {email.category}
              </span>
            )}
            {email.priority && (
              <span className={`badge ${getPriorityClass(email.priority)}`}>
                {email.priority === 'HIGH' && <FaExclamationCircle style={{ marginRight: '4px' }} />}
                {email.priority}
              </span>
            )}
            {email.priority_score && (
              <span className="badge badge-score">
                Score: {(email.priority_score * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default EmailList;
