"""
MailMate Dashboard API - Flask REST API with WebSocket support

This Flask application provides a comprehensive REST API for the MailMate React dashboard,
including endpoints for email management, AI analysis, real-time updates, and chat functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import uuid
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from email_loader import EmailDataLoader, EmailData
    from email_classifier import EmailClassifier
    from email_summarizer import EmailSummarizer
    from text_to_speech import TextToSpeech
    from data_export import DataExporter
    from session_manager import SessionManager, session_manager
except ImportError as e:
    print(f"Warning: Could not import backend modules: {e}")
    # Create mock classes for development
    class EmailData:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class EmailDataLoader:
        def generate_synthetic_emails(self, count=100):
            return []
    
    class EmailClassifier:
        def train(self, texts, labels):
            return {'test_accuracy': 0.95}
        
        def predict(self, text):
            from dataclasses import dataclass
            @dataclass
            class Result:
                predicted_category: str = "Work"
                confidence: float = 0.85
                probabilities: Dict[str, float] = None
            return Result()
    
    class EmailSummarizer:
        def summarize_email(self, text):
            from dataclasses import dataclass
            @dataclass
            class Result:
                success: bool = True
                summary: str = "This is a summary"
                processing_time: float = 1.0
            return Result()
    
    class TextToSpeech:
        def convert_text_to_speech(self, text, output_path):
            from dataclasses import dataclass
            @dataclass
            class Result:
                success: bool = True
                processing_time: float = 1.0
            return Result()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mailmate-dashboard-secret-key'
CORS(app, origins=["http://localhost:3000"])

# Socket.IO setup
socketio = SocketIO(
    app, 
    cors_allowed_origins="http://localhost:3000",
    async_mode='threading'
)

# Global variables for MailMate components
email_loader = EmailDataLoader()
email_classifier = EmailClassifier()
email_summarizer = EmailSummarizer()
text_to_speech = TextToSpeech()

# In-memory storage (replace with database in production)
emails_storage: List[Dict] = []
dashboard_stats: Dict = {}
chat_history: List[Dict] = []
active_connections: Dict[str, Dict] = {}

def generate_sample_emails():
    """Generate sample emails for demonstration."""
    global emails_storage
    
    try:
        # Generate synthetic emails
        sample_emails = email_loader.generate_synthetic_emails(count=50)
        
        emails_storage = []
        for email in sample_emails:
            email_dict = {
                'id': email.id,
                'from_address': email.from_address,
                'to_address': email.to_address,
                'cc_address': email.cc_address,
                'bcc_address': email.bcc_address,
                'subject': email.subject,
                'body': email.body,
                'timestamp': email.timestamp.isoformat() if hasattr(email.timestamp, 'isoformat') else str(email.timestamp),
                'category': email.category,
                'priority': email.priority,
                'attachments': email.attachments,
                'is_read': email.is_read,
                'folder': email.folder
            }
            emails_storage.append(email_dict)
        
        logger.info(f"Generated {len(emails_storage)} sample emails")
        
    except Exception as e:
        logger.error(f"Error generating sample emails: {e}")
        # Create fallback sample data
        emails_storage = [
            {
                'id': str(uuid.uuid4()),
                'from_address': 'john.doe@company.com',
                'to_address': 'user@mailmate.com',
                'cc_address': None,
                'bcc_address': None,
                'subject': 'Welcome to MailMate Dashboard',
                'body': 'This is a sample email to demonstrate the MailMate dashboard functionality.',
                'timestamp': datetime.now().isoformat(),
                'category': 'Work',
                'priority': 'Medium',
                'attachments': [],
                'is_read': False,
                'folder': 'INBOX'
            }
        ]

def calculate_dashboard_stats():
    """Calculate dashboard statistics."""
    global dashboard_stats
    
    total_emails = len(emails_storage)
    unread_emails = sum(1 for email in emails_storage if not email['is_read'])
    
    # Category distribution
    categories = {}
    for email in emails_storage:
        category = email['category']
        categories[category] = categories.get(category, 0) + 1
    
    # Priority distribution
    priorities = {}
    for email in emails_storage:
        priority = email['priority']
        priorities[priority] = priorities.get(priority, 0) + 1
    
    # Emails per day (last 7 days)
    now = datetime.now()
    emails_per_day = []
    for i in range(7):
        date = now - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        count = sum(1 for email in emails_storage 
                   if email['timestamp'].startswith(date_str))
        emails_per_day.append({
            'date': date_str,
            'count': count
        })
    emails_per_day.reverse()
    
    dashboard_stats = {
        'totalEmails': total_emails,
        'unreadEmails': unread_emails,
        'categoriesCount': len(categories),
        'priorityDistribution': priorities,
        'categoryDistribution': categories,
        'emailsPerDay': emails_per_day,
        'avgProcessingTime': 1.5,
        'lastUpdate': datetime.now().isoformat()
    }

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'data': {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/emails', methods=['GET'])
def get_emails():
    """Get emails with filtering and pagination."""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('pageSize', 20))
        sort_by = request.args.get('sortBy', 'timestamp')
        sort_order = request.args.get('sortOrder', 'desc')
        
        # Filters
        category = request.args.get('category')
        priority = request.args.get('priority')
        search = request.args.get('search')
        is_read = request.args.get('isRead')
        
        # Apply filters
        filtered_emails = emails_storage.copy()
        
        if category:
            filtered_emails = [e for e in filtered_emails if e['category'] == category]
        
        if priority:
            filtered_emails = [e for e in filtered_emails if e['priority'] == priority]
        
        if search:
            search_lower = search.lower()
            filtered_emails = [
                e for e in filtered_emails 
                if search_lower in e['subject'].lower() or 
                   search_lower in e['body'].lower() or
                   search_lower in e['from_address'].lower()
            ]
        
        if is_read is not None:
            is_read_bool = is_read.lower() == 'true'
            filtered_emails = [e for e in filtered_emails if e['is_read'] == is_read_bool]
        
        # Sort emails
        reverse = sort_order == 'desc'
        if sort_by == 'timestamp':
            filtered_emails.sort(key=lambda x: x['timestamp'], reverse=reverse)
        elif sort_by == 'subject':
            filtered_emails.sort(key=lambda x: x['subject'].lower(), reverse=reverse)
        elif sort_by == 'priority':
            priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
            filtered_emails.sort(
                key=lambda x: priority_order.get(x['priority'], 0), 
                reverse=reverse
            )
        
        # Pagination
        total = len(filtered_emails)
        start = (page - 1) * page_size
        end = start + page_size
        page_emails = filtered_emails[start:end]
        
        return jsonify({
            'success': True,
            'data': page_emails,
            'pagination': {
                'page': page,
                'pageSize': page_size,
                'total': total,
                'totalPages': (total + page_size - 1) // page_size
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/emails/<email_id>', methods=['GET'])
def get_email_by_id(email_id):
    """Get specific email by ID."""
    try:
        email = next((e for e in emails_storage if e['id'] == email_id), None)
        
        if not email:
            return jsonify({
                'success': False,
                'error': 'Email not found',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        return jsonify({
            'success': True,
            'data': email,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching email {email_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/emails/<email_id>/read', methods=['PATCH'])
def mark_email_read(email_id):
    """Mark email as read."""
    try:
        email = next((e for e in emails_storage if e['id'] == email_id), None)
        
        if not email:
            return jsonify({
                'success': False,
                'error': 'Email not found',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        email['is_read'] = True
        
        # Emit real-time update
        socketio.emit('email_updated', email)
        
        return jsonify({
            'success': True,
            'data': email,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error marking email as read: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/classify', methods=['POST'])
def classify_email():
    """Classify email text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Classify the email
        result = email_classifier.predict(text)
        
        classification_data = {
            'predicted_category': result.predicted_category,
            'confidence': result.confidence,
            'probabilities': result.probabilities or {}
        }
        
        return jsonify({
            'success': True,
            'data': classification_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error classifying email: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_email():
    """Summarize email text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Summarize the email
        result = email_summarizer.summarize_email(text)
        
        summary_data = {
            'success': result.success,
            'summary': result.summary,
            'processing_time': result.processing_time
        }
        
        return jsonify({
            'success': True,
            'data': summary_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error summarizing email: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats/dashboard', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics."""
    try:
        calculate_dashboard_stats()
        
        return jsonify({
            'success': True,
            'data': dashboard_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats/categories', methods=['GET'])
def get_category_distribution():
    """Get category distribution."""
    try:
        categories = {}
        for email in emails_storage:
            category = email['category']
            categories[category] = categories.get(category, 0) + 1
        
        return jsonify({
            'success': True,
            'data': categories,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching category distribution: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats/priorities', methods=['GET'])
def get_priority_distribution():
    """Get priority distribution."""
    try:
        priorities = {}
        for email in emails_storage:
            priority = email['priority']
            priorities[priority] = priorities.get(priority, 0) + 1
        
        return jsonify({
            'success': True,
            'data': priorities,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching priority distribution: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/chat/query', methods=['POST'])
def chat_query():
    """Handle AI chat queries."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        context = data.get('context', {})
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Process the query (simplified AI response)
        response_text = process_chat_query(query, context)
        
        # Store in chat history
        chat_entry = {
            'id': str(uuid.uuid4()),
            'query': query,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        chat_history.append(chat_entry)
        
        # Keep only last 100 entries
        if len(chat_history) > 100:
            chat_history.pop(0)
        
        ai_response = {
            'response': response_text,
            'type': 'text',
            'suggestions': [
                'Show me urgent emails',
                'What categories do I have?',
                'Generate a summary report'
            ]
        }
        
        return jsonify({
            'success': True,
            'data': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def process_chat_query(query: str, context: Dict) -> str:
    """Process natural language chat queries."""
    query_lower = query.lower()
    
    if 'urgent' in query_lower or 'high priority' in query_lower:
        urgent_count = sum(1 for e in emails_storage if e['priority'] == 'High')
        return f"You have {urgent_count} urgent/high priority emails in your inbox."
    
    elif 'categories' in query_lower or 'category' in query_lower:
        categories = set(e['category'] for e in emails_storage)
        return f"Your emails are categorized as: {', '.join(categories)}"
    
    elif 'unread' in query_lower:
        unread_count = sum(1 for e in emails_storage if not e['is_read'])
        return f"You have {unread_count} unread emails."
    
    elif 'summary' in query_lower or 'report' in query_lower:
        total = len(emails_storage)
        unread = sum(1 for e in emails_storage if not e['is_read'])
        categories = len(set(e['category'] for e in emails_storage))
        return f"Email Summary: {total} total emails, {unread} unread, across {categories} categories."
    
    elif 'help' in query_lower:
        return "I can help you with: checking urgent emails, viewing categories, counting unread emails, and generating summary reports. Just ask me in natural language!"
    
    else:
        return f"I understand you're asking about: '{query}'. I can help with email analysis, categories, priorities, and summaries. Try asking about urgent emails or unread messages!"

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    active_connections[request.sid] = {
        'connected_at': datetime.now().isoformat(),
        'rooms': []
    }
    
    # Send initial data
    emit('stats_updated', dashboard_stats)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")
    if request.sid in active_connections:
        del active_connections[request.sid]

@socketio.on('join_room')
def handle_join_room(data):
    """Handle joining a room for targeted updates."""
    room = data.get('room')
    if room:
        join_room(room)
        if request.sid in active_connections:
            active_connections[request.sid]['rooms'].append(room)
        logger.info(f"Client {request.sid} joined room: {room}")

@socketio.on('leave_room')
def handle_leave_room(data):
    """Handle leaving a room."""
    room = data.get('room')
    if room:
        leave_room(room)
        if request.sid in active_connections:
            rooms = active_connections[request.sid]['rooms']
            if room in rooms:
                rooms.remove(room)
        logger.info(f"Client {request.sid} left room: {room}")

@socketio.on('subscribe_dashboard')
def handle_subscribe_dashboard():
    """Subscribe to dashboard updates."""
    join_room('dashboard')
    logger.info(f"Client {request.sid} subscribed to dashboard updates")

@socketio.on('ping')
def handle_ping(timestamp):
    """Handle ping for latency testing."""
    emit('pong', timestamp)

def background_stats_updater():
    """Background task to update statistics periodically."""
    while True:
        try:
            time.sleep(30)  # Update every 30 seconds
            calculate_dashboard_stats()
            socketio.emit('stats_updated', dashboard_stats, room='dashboard')
            logger.info("Dashboard stats updated via WebSocket")
        except Exception as e:
            logger.error(f"Error in background stats updater: {e}")

def simulate_new_email():
    """Simulate receiving new emails for demonstration."""
    while True:
        try:
            time.sleep(60)  # New email every minute
            
            # Create a new sample email
            new_email = {
                'id': str(uuid.uuid4()),
                'from_address': f'sender{len(emails_storage)}@example.com',
                'to_address': 'user@mailmate.com',
                'cc_address': None,
                'bcc_address': None,
                'subject': f'New Message #{len(emails_storage) + 1}',
                'body': 'This is a simulated new email for real-time demonstration.',
                'timestamp': datetime.now().isoformat(),
                'category': 'Work',
                'priority': 'Medium',
                'attachments': [],
                'is_read': False,
                'folder': 'INBOX'
            }
            
            emails_storage.append(new_email)
            
            # Emit real-time update
            socketio.emit('email_received', new_email)
            
            # Update stats
            calculate_dashboard_stats()
            socketio.emit('stats_updated', dashboard_stats)
            
            logger.info(f"Simulated new email: {new_email['subject']}")
            
        except Exception as e:
            logger.error(f"Error simulating new email: {e}")

# Data Export Endpoints

@app.route('/api/export/emails/<format_type>', methods=['POST'])
def export_emails(format_type):
    """Export email data in specified format."""
    try:
        if format_type.lower() not in ['csv', 'json']:
            return jsonify({'error': 'Unsupported format. Use CSV or JSON'}), 400
        
        data = request.get_json() or {}
        filters = data.get('filters', {})
        include_metadata = data.get('include_metadata', True)
        filename = data.get('filename')
        
        # Initialize exporter
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from data_export import DataExporter
            exporter = DataExporter()
        except ImportError as e:
            logger.warning(f"Export module not available: {e}")
            return jsonify({'error': 'Export functionality not available'}), 503
        
        # Apply filters to emails
        filtered_emails = list(emails_storage)
        
        if filters.get('categories'):
            categories = filters['categories'] if isinstance(filters['categories'], list) else [filters['categories']]
            filtered_emails = [e for e in filtered_emails if e.get('category') in categories]
        
        if filters.get('priorities'):
            priorities = filters['priorities'] if isinstance(filters['priorities'], list) else [filters['priorities']]
            filtered_emails = [e for e in filtered_emails if e.get('priority') in priorities]
        
        if filters.get('search'):
            search_term = filters['search'].lower()
            filtered_emails = [
                e for e in filtered_emails 
                if search_term in e.get('subject', '').lower() or 
                   search_term in e.get('body', '').lower()
            ]
        
        # Export based on format
        if format_type.lower() == 'csv':
            filepath = exporter.export_emails_to_csv(
                filtered_emails,
                filename=filename,
                include_metadata=include_metadata,
                filters=filters
            )
        else:  # JSON
            filepath = exporter.export_emails_to_json(
                filtered_emails,
                filename=filename,
                include_metadata=include_metadata,
                filters=filters
            )
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'record_count': len(filtered_emails),
            'format': format_type.upper(),
            'export_date': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/export/analytics', methods=['POST'])
def export_analytics():
    """Export analytics data."""
    try:
        data = request.get_json() or {}
        format_type = data.get('format', 'JSON')
        include_charts = data.get('include_charts', True)
        filename = data.get('filename')
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from data_export import DataExporter
            exporter = DataExporter()
        except ImportError as e:
            logger.warning(f"Export module not available: {e}")
            return jsonify({'error': 'Export functionality not available'}), 503
        
        # Generate analytics data
        analytics_data = {
            'dashboard_stats': dashboard_stats,
            'email_analytics': {
                'total_emails': len(emails_storage),
                'categories': {},
                'priorities': {},
                'daily_volume': {},
                'response_times': []
            },
            'performance_metrics': {
                'api_calls': len(chat_history),
                'export_operations': 0,
                'uptime': str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0))
            }
        }
        
        # Calculate category and priority distributions
        for email in emails_storage:
            cat = email.get('category', 'Unknown')
            pri = email.get('priority', 'Medium')
            analytics_data['email_analytics']['categories'][cat] = analytics_data['email_analytics']['categories'].get(cat, 0) + 1
            analytics_data['email_analytics']['priorities'][pri] = analytics_data['email_analytics']['priorities'].get(pri, 0) + 1
        
        filepath = exporter.export_analytics_report(
            analytics_data,
            format_type=format_type,
            filename=filename,
            include_charts_data=include_charts
        )
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'format': format_type.upper(),
            'export_date': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analytics export error: {e}")
        return jsonify({'error': f'Analytics export failed: {str(e)}'}), 500


@app.route('/api/export/bulk', methods=['POST'])
def export_bulk():
    """Create comprehensive bulk export."""
    try:
        data = request.get_json() or {}
        filename = data.get('filename')
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from data_export import DataExporter
            exporter = DataExporter()
        except ImportError as e:
            logger.warning(f"Export module not available: {e}")
            return jsonify({'error': 'Export functionality not available'}), 503
        
        # Prepare analytics data
        analytics_data = {
            'total_emails': len(emails_storage),
            'dashboard_stats': dashboard_stats,
            'categories': {},
            'priorities': {}
        }
        
        for email in emails_storage:
            cat = email.get('category', 'Unknown')
            pri = email.get('priority', 'Medium')
            analytics_data['categories'][cat] = analytics_data['categories'].get(cat, 0) + 1
            analytics_data['priorities'][pri] = analytics_data['priorities'].get(pri, 0) + 1
        
        # Mock summaries (in real implementation, would get from email summarizer)
        summaries = [
            {
                'email_id': email.get('id', f'email_{i}'),
                'summary': f'Summary for {email.get("subject", "email")}',
                'category': email.get('category', 'Unknown'),
                'sentiment': 'neutral',
                'key_points': ['Point 1', 'Point 2']
            }
            for i, email in enumerate(emails_storage[:10])  # Limit for demo
        ]
        
        filepath = exporter.create_bulk_export(
            list(emails_storage),
            analytics_data,
            summaries,
            filename=filename
        )
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'export_type': 'BULK',
            'record_count': len(emails_storage),
            'export_date': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Bulk export error: {e}")
        return jsonify({'error': f'Bulk export failed: {str(e)}'}), 500


# Session Management Endpoints

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create a new dashboard session."""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback session creation
            session_id = str(uuid.uuid4())
            return jsonify({
                'success': True,
                'session_id': session_id,
                'user_id': user_id,
                'created_at': datetime.now().isoformat()
            })
        
        state = session_manager.create_session(user_id)
        
        return jsonify({
            'success': True,
            'session_id': state.session_id,
            'user_id': state.user_id,
            'created_at': state.last_updated.isoformat(),
            'expires_at': state.expires_at.isoformat() if state.expires_at else None
        })
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({'error': f'Session creation failed: {str(e)}'}), 500


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session state."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
            state = session_manager.get_session(session_id)
            
            if not state:
                return jsonify({'error': 'Session not found or expired'}), 404
            
            return jsonify({
                'success': True,
                'session': session_manager.get_browser_storage_state(session_id)
            })
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback response
            return jsonify({
                'success': True,
                'session': {
                    'sessionId': session_id,
                    'userId': 'default',
                    'emailFilters': {},
                    'viewPreferences': {},
                    'dashboardLayout': {},
                    'recentSearches': [],
                    'bookmarkedEmails': [],
                    'notificationSettings': {},
                    'themeSettings': {},
                    'lastUpdated': datetime.now().isoformat()
                }
            })
        
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        return jsonify({'error': f'Session retrieval failed: {str(e)}'}), 500


@app.route('/api/session/<session_id>', methods=['PUT'])
def update_session(session_id):
    """Update session state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
            
            # Update different parts of session state
            if 'emailFilters' in data:
                session_manager.update_email_filters(session_id, data['emailFilters'])
            
            if 'viewPreferences' in data:
                session_manager.update_view_preferences(session_id, data['viewPreferences'])
            
            if 'dashboardLayout' in data:
                session_manager.update_dashboard_layout(session_id, data['dashboardLayout'])
            
            if 'themeSettings' in data:
                session_manager.update_theme_settings(session_id, data['themeSettings'])
            
            if 'recentSearch' in data:
                session_manager.add_recent_search(session_id, data['recentSearch'])
            
            if 'bookmarkToggle' in data:
                session_manager.toggle_email_bookmark(session_id, data['bookmarkToggle'])
            
            return jsonify({
                'success': True,
                'updated_at': datetime.now().isoformat()
            })
            
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback success response
            return jsonify({
                'success': True,
                'updated_at': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Session update error: {e}")
        return jsonify({'error': f'Session update failed: {str(e)}'}), 500


@app.route('/api/session/<session_id>/extend', methods=['POST'])
def extend_session(session_id):
    """Extend session expiration."""
    try:
        data = request.get_json() or {}
        additional_time = data.get('additional_time')
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
            success = session_manager.extend_session(session_id, additional_time)
            
            if not success:
                return jsonify({'error': 'Session not found'}), 404
            
            state = session_manager.get_session(session_id)
            return jsonify({
                'success': True,
                'expires_at': state.expires_at.isoformat() if state.expires_at else None
            })
            
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback response
            return jsonify({
                'success': True,
                'expires_at': (datetime.now() + timedelta(days=1)).isoformat()
            })
        
    except Exception as e:
        logger.error(f"Session extension error: {e}")
        return jsonify({'error': f'Session extension failed: {str(e)}'}), 500


@app.route('/api/session/<session_id>/sync', methods=['POST'])
def sync_session(session_id):
    """Synchronize session with browser storage."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No browser state provided'}), 400
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
            success = session_manager.sync_from_browser_storage(session_id, data)
            
            if not success:
                return jsonify({'error': 'Session not found'}), 404
            
            return jsonify({
                'success': True,
                'synced_at': datetime.now().isoformat()
            })
            
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback response
            return jsonify({
                'success': True,
                'synced_at': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Session sync error: {e}")
        return jsonify({'error': f'Session sync failed: {str(e)}'}), 500


@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
            success = session_manager.delete_session(session_id)
            
            return jsonify({
                'success': success,
                'deleted_at': datetime.now().isoformat()
            })
            
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback response
            return jsonify({
                'success': True,
                'deleted_at': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        return jsonify({'error': f'Session deletion failed: {str(e)}'}), 500


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List active sessions."""
    try:
        user_id = request.args.get('user_id')
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        try:
            from session_manager import session_manager
            sessions = session_manager.list_active_sessions(user_id)
            
            return jsonify({
                'success': True,
                'sessions': sessions,
                'total_count': len(sessions)
            })
            
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Fallback response
            return jsonify({
                'success': True,
                'sessions': [],
                'total_count': 0
            })
        
    except Exception as e:
        logger.error(f"Session listing error: {e}")
        return jsonify({'error': f'Session listing failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Initialize data
    generate_sample_emails()
    calculate_dashboard_stats()
    
    # Start background tasks
    stats_thread = threading.Thread(target=background_stats_updater, daemon=True)
    stats_thread.start()
    
    email_simulation_thread = threading.Thread(target=simulate_new_email, daemon=True)
    email_simulation_thread.start()
    
    logger.info("Starting MailMate Dashboard API server...")
    logger.info(f"Generated {len(emails_storage)} sample emails")
    
    # Start the Flask-SocketIO server
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        use_reloader=False  # Disable reloader to prevent duplicate background threads
    )