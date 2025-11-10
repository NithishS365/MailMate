#!/usr/bin/env python3
"""
MailMate Production API Server

This is the production-ready version of MailMate with all core features:
- Data Export (CSV/JSON)
- Session Management
- Email Processing (without heavy transformer models)
- Extensible Architecture
- Real-time API endpoints

Usage: python mailmate_server.py
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json
import random

# Add paths to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, 'backend')
sys.path.extend([current_dir, backend_dir])

# Configure Python path for imports
os.environ['PYTHONPATH'] = os.pathsep.join([current_dir, backend_dir, os.environ.get('PYTHONPATH', '')])

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# Import MailMate core modules
from data_export import DataExporter
from session_manager import SessionManager
from email_loader import EmailDataLoader

# Gmail OAuth2 configuration
GMAIL_CLIENT_SECRET_FILE = 'config/client_secret.json'
GMAIL_TOKEN_FILE = 'config/token.pickle'
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("mailmate_server")

# Initialize Flask app and Socket.IO
from flask_socketio import SocketIO

# If a React build exists in frontend/build, serve static files from there.
FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
if os.path.exists(FRONTEND_BUILD):
    app = Flask(__name__, static_folder=FRONTEND_BUILD, static_url_path='')
    logger.info(f"Serving frontend build from {FRONTEND_BUILD}")
else:
    app = Flask(__name__)

# Configure CORS for both REST and WebSocket
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "expose_headers": ["Content-Range", "X-Content-Range"]
    }
})

# Initialize Socket.IO with CORS support
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"], async_mode='threading')

# Initialize MailMate components
try:
    logger.info("🚀 Initializing MailMate components...")
    
    data_exporter = DataExporter()
    logger.info("✅ DataExporter initialized")
    
    session_manager = SessionManager()
    logger.info("✅ SessionManager initialized")
    
    email_loader = EmailDataLoader()
    logger.info("✅ EmailLoader initialized")
    
    # Load emails - prefer scraped data over synthetic
    sample_emails = email_loader.load_emails_smart(
        prefer_scraped=True, 
        data_dir='data',
        fallback_count=50
    )
    logger.info(f"✅ Loaded {len(sample_emails)} emails")
    
    # Test extensible interfaces
    interface_status = {}
    try:
        from interfaces import BaseMLModel, ConfigurationManager
        interface_status["core_interfaces"] = "✅ Available"
        logger.info("✅ Extensible interfaces loaded")
    except Exception as e:
        interface_status["core_interfaces"] = f"⚠️ Limited: {str(e)}"
        logger.warning(f"⚠️ Extensible interfaces limited: {e}")
    
    logger.info("🎉 MailMate initialization complete!")
    
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    sample_emails = []
    interface_status = {"error": str(e)}

# Helper utilities to work with EmailData objects or plain dicts
def _email_get(obj, key, default=None):
    """Safely get a field from an email which may be an EmailData or a dict."""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        # dataclass or object with to_dict
        if hasattr(obj, 'to_dict'):
            try:
                return obj.to_dict().get(key, default)
            except Exception:
                pass
        return getattr(obj, key, default)
    except Exception:
        return default


def _email_to_dict(obj):
    """Convert EmailData or dict-like email into a plain dict suitable for JSON."""
    try:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # fallback to vars()
        return dict(vars(obj))
    except Exception:
        return {}

# ============================================================================
# Dashboard UI
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MailMate Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 30px;
        }
        .status-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #007bff;
        }
        .status-card.success {
            border-left-color: #28a745;
        }
        .status-card.warning {
            border-left-color: #ffc107;
        }
        .status-card.info {
            border-left-color: #17a2b8;
        }
        .status-card h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .api-section {
            padding: 30px;
            border-top: 1px solid #eee;
        }
        .endpoint {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
        }
        .endpoint-method {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
        }
        .endpoint-method.post {
            background: #28a745;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .footer {
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
        }
        .demo-section {
            padding: 30px;
            background: #f8f9fa;
        }
        .demo-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        .output {
            background: #000;
            color: #0f0;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin-top: 15px;
            max-height: 300px;
            overflow-y: auto;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 MailMate Dashboard</h1>
            <p>Production-Ready Email Management Platform</p>
            <p><strong>Server Time:</strong> {{ timestamp }}</p>
        </div>

        <div class="status-grid">
            <div class="status-card success">
                <h3>✅ System Status</h3>
                <p><strong>Status:</strong> Running</p>
                <p><strong>Sample Emails:</strong> {{ email_count }}</p>
                <p><strong>Active Sessions:</strong> {{ session_count }}</p>
            </div>

            <div class="status-card success">
                <h3>🏗️ Core Features</h3>
                <p>✅ Data Export (CSV/JSON)</p>
                <p>✅ Session Management</p>
                <p>✅ Email Processing</p>
                <p>✅ Real-time API</p>
            </div>

            <div class="status-card info">
                <h3>🔌 Extensible Architecture</h3>
                <p>✅ ML Model Interfaces</p>
                <p>✅ OAuth2 Email Providers</p>
                <p>✅ Plugin System</p>
                <p>✅ Configuration Management</p>
            </div>

            <div class="status-card warning">
                <h3>⚡ Optional Features</h3>
                <p>⚠️ Batch Processing (Celery/Redis)</p>
                <p>⚠️ Heavy ML Models (Transformers)</p>
                <p>✅ Lightweight Processing</p>
            </div>
        </div>

        <div class="demo-section">
            <h2>🧪 Live API Demo</h2>
            <p>Test the MailMate API endpoints directly from the dashboard:</p>
            
            <div class="demo-buttons">
                <button class="btn" onclick="testEndpoint('/api/status')">System Status</button>
                <button class="btn" onclick="testEndpoint('/api/emails?per_page=5')">Get Emails</button>
                <button class="btn" onclick="testExport('csv')">Export CSV</button>
                <button class="btn" onclick="testExport('json')">Export JSON</button>
                <button class="btn" onclick="testSession()">Create Session</button>
                <button class="btn" onclick="testEndpoint('/api/interfaces/status')">Interface Status</button>
            </div>
            
            <div id="output" class="output"></div>
        </div>

        <div class="api-section">
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="endpoint-method">GET</span>
                <strong>/api/status</strong> - System status and component information
            </div>
            
            <div class="endpoint">
                <span class="endpoint-method">GET</span>
                <strong>/api/emails</strong> - List sample emails (supports pagination)
            </div>
            
            <div class="endpoint">
                <span class="endpoint-method post">POST</span>
                <strong>/api/export/emails/csv</strong> - Export emails to CSV format
            </div>
            
            <div class="endpoint">
                <span class="endpoint-method post">POST</span>
                <strong>/api/export/emails/json</strong> - Export emails to JSON format
            </div>
            
            <div class="endpoint">
                <span class="endpoint-method post">POST</span>
                <strong>/api/session/create</strong> - Create new dashboard session
            </div>
            
            <div class="endpoint">
                <span class="endpoint-method">GET</span>
                <strong>/api/session/{id}</strong> - Get session state
            </div>
            
            <div class="endpoint">
                <span class="endpoint-method">GET</span>
                <strong>/api/interfaces/status</strong> - Extensible interfaces status
            </div>
        </div>

        <div class="footer">
            <p>&copy; 2025 MailMate - Enterprise Email Management Platform</p>
            <p>Built with Flask, React, and Extensible Architecture</p>
        </div>
    </div>

    <script>
        function testEndpoint(url) {
            const output = document.getElementById('output');
            output.style.display = 'block';
            output.innerHTML = `> Testing ${url}...\\n`;
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    output.innerHTML += `✅ Success:\\n${JSON.stringify(data, null, 2)}\\n\\n`;
                })
                .catch(error => {
                    output.innerHTML += `❌ Error: ${error}\\n\\n`;
                });
        }
        
        function testExport(format) {
            const output = document.getElementById('output');
            output.style.display = 'block';
            output.innerHTML = `> Testing ${format.toUpperCase()} export...\\n`;
            
            fetch(`/api/export/emails/${format}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: `demo_export_${format}`,
                    include_metadata: true
                })
            })
                .then(response => response.json())
                .then(data => {
                    output.innerHTML += `✅ Export Success:\\n${JSON.stringify(data, null, 2)}\\n\\n`;
                })
                .catch(error => {
                    output.innerHTML += `❌ Export Error: ${error}\\n\\n`;
                });
        }
        
        function testSession() {
            const output = document.getElementById('output');
            output.style.display = 'block';
            output.innerHTML = `> Creating session...\\n`;
            
            fetch('/api/session/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: 'demo_user_' + Date.now()
                })
            })
                .then(response => response.json())
                .then(data => {
                    output.innerHTML += `✅ Session Created:\\n${JSON.stringify(data, null, 2)}\\n\\n`;
                })
                .catch(error => {
                    output.innerHTML += `❌ Session Error: ${error}\\n\\n`;
                });
        }
    </script>
</body>
</html>
"""

@app.route('/demo-dashboard')
def demo_dashboard():
    """Demo dashboard kept for quick local testing. Production should use the React frontend build."""
    try:
        # Get session count safely
        session_count = 0
        if hasattr(session_manager, '_session_cache'):
            session_count = len(session_manager._session_cache)
        elif hasattr(session_manager, 'list_active_sessions'):
            session_count = len(session_manager.list_active_sessions())
        
        return render_template_string(
            DASHBOARD_HTML,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            email_count=len(sample_emails),
            session_count=session_count
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template_string(
            DASHBOARD_HTML,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            email_count=len(sample_emails),
            session_count=0
        )


# Catch-all to serve React frontend build if available; otherwise show demo dashboard
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    try:
        if os.path.exists(FRONTEND_BUILD):
            full_path = os.path.join(FRONTEND_BUILD, path)
            if path != '' and os.path.exists(full_path):
                return app.send_static_file(path)
            return app.send_static_file('index.html')
        return demo_dashboard()
    except Exception as e:
        logger.error(f"Failed to serve frontend: {e}")
        return demo_dashboard()

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/status')
def api_status():
    """Get comprehensive system status."""
    try:
        # Get session count safely
        session_count = 0
        if hasattr(session_manager, '_session_cache'):
            session_count = len(session_manager._session_cache)
        elif hasattr(session_manager, 'list_active_sessions'):
            session_count = len(session_manager.list_active_sessions())
        
        return jsonify({
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "data_exporter": "✅ Ready",
                "session_manager": "✅ Ready",
                "email_loader": "✅ Ready",
                "sample_emails": len(sample_emails),
                "active_sessions": session_count
            },
            "extensible_interfaces": interface_status,
            "features": {
                "data_export": "✅ CSV/JSON export with metadata",
                "session_management": "✅ Browser storage sync",
                "email_processing": "✅ Classification and filtering",
                "real_time_api": "✅ REST endpoints with CORS",
                "extensible_architecture": "✅ Plugin and provider framework"
            }
        })
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emails')
def get_emails():
    """Get paginated list of emails."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        category = request.args.get('category')
        
        # Filter by category if specified (support EmailData or dict)
        filtered_emails = sample_emails
        if category:
            filtered_emails = [e for e in sample_emails if (_email_get(e, 'category') or '').lower() == category.lower()]
        
        # Pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        emails_page = filtered_emails[start_idx:end_idx]
        # Convert to plain dicts for JSON serialization
        emails_page_dicts = [_email_to_dict(e) for e in emails_page]

        return jsonify({
            "emails": emails_page_dicts,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": len(filtered_emails),
                "total_pages": (len(filtered_emails) + per_page - 1) // per_page
            },
            "categories": list({(_email_get(e, 'category') or 'Unknown') for e in sample_emails})
        })
    except Exception as e:
        logger.error(f"Failed to get emails: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/export/emails/csv', methods=['POST'])
def export_emails_csv():
    """Export emails to CSV format."""
    try:
        data = request.get_json() or {}
        filename = data.get('filename', 'mailmate_emails')
        include_metadata = data.get('include_metadata', True)
        
        # Apply filters
        emails_to_export = sample_emails
        if 'filters' in data and data['filters']:
            filters = data['filters']
            if 'categories' in filters:
                emails_to_export = [e for e in emails_to_export if _email_get(e, 'category') in filters['categories']]
        
        # Export to CSV
        csv_file = data_exporter.export_emails_to_csv(
            emails_to_export,
            filename=filename,
            include_metadata=include_metadata
        )
        
        return jsonify({
            "status": "success",
            "file_path": csv_file,
            "records_exported": len(emails_to_export),
            "export_type": "CSV",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/export/emails/json', methods=['POST'])
def export_emails_json():
    """Export emails to JSON format."""
    try:
        data = request.get_json() or {}
        filename = data.get('filename', 'mailmate_emails')
        include_metadata = data.get('include_metadata', True)
        
        # Apply filters
        emails_to_export = sample_emails
        if 'filters' in data and data['filters']:
            filters = data['filters']
            if 'categories' in filters:
                emails_to_export = [e for e in emails_to_export if _email_get(e, 'category') in filters['categories']]
        
        # Export to JSON

        json_file = data_exporter.export_emails_to_json(
            emails_to_export,
            filename=filename,
            include_metadata=include_metadata
        )
        
        return jsonify({
            "status": "success",
            "file_path": json_file,
            "records_exported": len(emails_to_export),
            "export_type": "JSON",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create new dashboard session."""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', f'user_{datetime.now().timestamp():.0f}')
        
        session_state = session_manager.create_session(user_id)
        
        return jsonify({
            "status": "success",
            "session_id": session_state.session_id,
            "user_id": session_state.user_id,
            "expires_at": session_state.expires_at.isoformat(),
            "created_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/<session_id>')
def get_session(session_id):
    """Get session state."""
    try:
        session_state = session_manager.get_session(session_id)
        
        if not session_state:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify({
            "session_id": session_state.session_id,
            "user_id": session_state.user_id,
            "email_filters": session_state.email_filters,
            "view_preferences": session_state.view_preferences,
            "dashboard_layout": session_state.dashboard_layout,
            "last_updated": session_state.last_updated.isoformat(),
            "expires_at": session_state.expires_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats/dashboard')
def dashboard_stats():
    """Get dashboard statistics."""
    try:
        return jsonify({
            "emailCount": len(sample_emails),
            "emailsByCategory": {
                "inbox": sum(1 for e in sample_emails if _email_get(e, 'category') == 'inbox'),
                "sent": sum(1 for e in sample_emails if _email_get(e, 'category') == 'sent'),
                "draft": sum(1 for e in sample_emails if _email_get(e, 'category') == 'draft'),
                "spam": sum(1 for e in sample_emails if _email_get(e, 'category') == 'spam'),
            },
            "emailsByPriority": {
                "high": sum(1 for e in sample_emails if _email_get(e, 'priority') == 'high'),
                "medium": sum(1 for e in sample_emails if _email_get(e, 'priority') == 'medium'),
                "low": sum(1 for e in sample_emails if _email_get(e, 'priority') == 'low'),
            },
            "unreadCount": sum(1 for e in sample_emails if not _email_get(e, 'read', False)),
            "lastUpdateTime": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/emails')
def email_analytics():
    """Get email analytics data."""
    try:
        time_range = request.args.get('timeRange', '30d')
        days = int(time_range.rstrip('d'))
        
        # Generate some mock analytics data
        now = datetime.now()
        data = []
        for i in range(days):
            date = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            data.append({
                "date": date,
                "inbound": round(random.random() * 10),
                "outbound": round(random.random() * 8),
                "spam": round(random.random() * 2)
            })
        
        return jsonify({
            "timeRange": time_range,
            "data": data,
            "summary": {
                "totalInbound": sum(d["inbound"] for d in data),
                "totalOutbound": sum(d["outbound"] for d in data),
                "totalSpam": sum(d["spam"] for d in data),
                "averageResponseTime": round(random.random() * 60, 1)  # minutes
            }
        })
    except Exception as e:
        logger.error(f"Failed to get email analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/email/connect/gmail', methods=['POST'])
def connect_gmail():
    """Connect to Gmail using OAuth2."""
    try:
        data = request.get_json() or {}
        email_address = data.get('email_address')
        
        if not email_address:
            return jsonify({"error": "Email address required"}), 400
            
        # Create new EmailLoader instance for this connection
        loader = EmailDataLoader()
        
        try:
            loader.connect_imap(
                email_address=email_address,
                provider='gmail',
                use_oauth2=True,
                client_secret_file=GMAIL_CLIENT_SECRET_FILE,
                token_file=GMAIL_TOKEN_FILE
            )
            
            # Fetch some emails to verify connection
            emails = loader.fetch_emails_imap(folder='INBOX', limit=10)
            
            return jsonify({
                "status": "success",
                "message": f"Successfully connected to Gmail for {email_address}",
                "email_count": len(emails),
                "folders": loader.get_folder_list()
            })
            
        finally:
            loader.disconnect_imap()
            
    except Exception as e:
        logger.error(f"Gmail connection failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/email/sync/gmail', methods=['POST'])
def sync_gmail():
    """Sync emails from Gmail."""
    try:
        data = request.get_json() or {}
        email_address = data.get('email_address')
        folder = data.get('folder', 'INBOX')
        limit = int(data.get('limit', 100))
        
        if not email_address:
            return jsonify({"error": "Email address required"}), 400
            
        # Create new EmailLoader instance for this sync
        loader = EmailDataLoader()
        
        try:
            loader.connect_imap(
                email_address=email_address,
                provider='gmail',
                use_oauth2=True,
                client_secret_file=GMAIL_CLIENT_SECRET_FILE,
                token_file=GMAIL_TOKEN_FILE
            )
            
            # Fetch emails
            emails = loader.fetch_emails_imap(folder=folder, limit=limit)
            
            # Convert to dicts for JSON response
            email_dicts = [
                {
                    'id': e.id,
                    'subject': e.subject,
                    'from': e.from_address,
                    'to': e.to_address,
                    'date': e.timestamp.isoformat() if e.timestamp else None,
                    'folder': e.folder,
                    'has_attachments': bool(e.attachments)
                }
                for e in emails
            ]
            
            return jsonify({
                "status": "success",
                "folder": folder,
                "emails": email_dicts,
                "count": len(emails)
            })
            
        finally:
            loader.disconnect_imap()
            
    except Exception as e:
        logger.error(f"Gmail sync failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/interfaces/status')
def interfaces_status():
    """Get extensible interfaces status."""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "interfaces": {}
        }
        
        # Test each interface
        interface_tests = [
            ("ml_models", "interfaces.ml_models", "BaseMLModel"),
            ("email_providers", "interfaces.email_providers", "OAuth2Provider"),
            ("plugin_system", "interfaces.plugin_system", "PluginManager"),
            ("configuration", "interfaces.configuration", "ConfigurationManager"),
            ("batch_processing", "interfaces.batch_processing", "BatchProcessor")
        ]
        
        for name, module_name, class_name in interface_tests:
            try:
                __import__(module_name)
                status["interfaces"][name] = {
                    "status": "✅ Available",
                    "description": f"{class_name} interface ready for extension"
                }
            except ImportError as e:
                status["interfaces"][name] = {
                    "status": "⚠️ Limited",
                    "error": str(e),
                    "solution": "Install missing dependencies"
                }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Interface status check failed: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "available_endpoints": [
        "GET /",
        "GET /api/status",
        "GET /api/emails",
        "POST /api/export/emails/csv",
        "POST /api/export/emails/json",
        "POST /api/session/create",
        "GET /api/session/{id}",
        "GET /api/interfaces/status"
    ]}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# Main Application
# ============================================================================

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('error')
def handle_error(error):
    logger.error(f"WebSocket error: {error}")

if __name__ == '__main__':
    logger.info("🚀 Starting MailMate Production Server...")
    logger.info(f"📊 System Ready with {len(sample_emails)} emails")
    logger.info("🌐 Dashboard: http://127.0.0.1:5000")
    logger.info("📡 API Base: http://127.0.0.1:5000/api")
    
    try:
        socketio.run(
            app,
            host='127.0.0.1',  # Only listen on localhost for development
            port=5000,
            debug=True,  # Enable debug mode for development
            use_reloader=False  # Disable reloader to prevent double startup
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise