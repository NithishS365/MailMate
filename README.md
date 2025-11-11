# MailMate - AI-Powered Email Management System

![MailMate](https://img.shields.io/badge/Python-3.13+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![React](https://img.shields.io/badge/React-18.2+-blue.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange.svg)

MailMate is a comprehensive data science project that demonstrates modern email management using machine learning, advanced analytics, and professional UI design. It features AI-generated emails, ML-based classification and importance scoring, bulk operations, advanced search with multiple filters, and an interactive analytics dashboard with professional GitHub-inspired design.

## 🌟 Features

### Backend (FastAPI + Python)
- **AI Email Generation**: Generates 50 realistic professional emails across different categories (Work, Personal, Urgent, Promotion)
- **Advanced Search**: Full-text search with 7 filter parameters (query, categories, priorities, read status, date range, min priority score)
- **Bulk Operations**: Mark multiple emails as read/unread, bulk delete with single API call
- **Importance Scoring**: ML-based importance algorithm with 6 factors (recency, read status, sender frequency, content length, attachments, category urgency)
- **Data Cleaning**: Uses Pandas for preprocessing and cleaning email data
- **ML Classification**: Implements Naive Bayes classifier for categorizing and prioritizing emails
- **RESTful API**: Complete CRUD operations plus advanced endpoints for email management
- **MySQL Database**: Persistent storage with SQLAlchemy ORM
- **Analytics**: Advanced statistical analysis with time-series data and comprehensive metrics
- **Clean Logging**: Optimized console output without SQL query spam

### Frontend (React + Recharts)
- **Professional UI Design**: GitHub-inspired blue/gray color scheme with minimal shadows and clean borders
- **Advanced Search**: Collapsible filter panel with multi-select for categories/priorities, date range picker, and priority score filtering
- **Email Quick Actions**: Hover-reveal buttons for star/unstar, archive, and delete with confirmation dialogs
- **Visual Indicators**: Unread dot with pulse animation, color-coded priority badges
- **Modern Dashboard**: Clean, responsive UI with borderless stat cards and bottom accent bars
- **Email Management**: View, search, filter emails by multiple criteria with advanced search
- **Data Visualization**: 10+ interactive charts (bar, line, area, pie) showing email distribution and trends
- **Real-time Statistics**: Live metrics with trend indicators (📈/📉) and color-coded percentages
- **Daily Digest**: Top 5 priority emails with TTS functionality
- **Responsive Design**: Works seamlessly on desktop and mobile

### ML & Data Science
- **Text Classification**: TF-IDF vectorization with Naive Bayes/Logistic Regression
- **Priority Scoring**: Automated email prioritization based on content
- **Importance Algorithm**: Multi-factor scoring (0-100) with confidence recommendations (CRITICAL/HIGH/MEDIUM/LOW/MINIMAL)
- **Feature Engineering**: Extracts urgency, promotional, and work-related indicators
- **Data Preprocessing**: Stop word removal, text cleaning, and normalization
- **Recency Scoring**: Time-based scoring with 10-point scale for email freshness
- **Sender Analysis**: Frequency-based sender importance calculation

## 📋 Prerequisites

- Python 3.13 or higher
- Node.js 16 or higher
- MySQL 8.0 or higher
- npm or yarn
- Git (optional, for version control)

## 🚀 Installation & Setup

### 1. Clone or Download the Project

```powershell
cd K:\M\MailMate
```

### 2. Database Setup

Install and start MySQL, then create the database:

```sql
CREATE DATABASE mailmate CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Or via command line:

```powershell
mysql -u root -p -e "CREATE DATABASE mailmate CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

### 3. Backend Setup

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment file
copy .env.example .env
# Edit .env with your MySQL credentials
```

Edit the `.env` file with your MySQL credentials:
```env
DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/mailmate
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=root
DATABASE_PASSWORD=YOUR_PASSWORD
DATABASE_NAME=mailmate
```

### 4. Frontend Setup

```powershell
# Open a new terminal and navigate to frontend directory
cd K:\M\MailMate\frontend

# Install dependencies
npm install
```

## 🎮 Running the Application

### 1. Start the Backend Server

```powershell
cd K:\M\MailMate\backend
.\venv\Scripts\Activate
python main.py
```

The backend will run on: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### 2. Start the Frontend Development Server

```powershell
# In a new terminal
cd K:\M\MailMate\frontend
npm run dev
```

The frontend will run on: `http://localhost:3000`

## 📸 Screenshots & Demo

### Main Dashboard
- Clean GitHub-inspired UI with blue/gray color scheme
- Borderless stat cards with hover accent bars
- Unread email indicators with pulse animation

### Advanced Search
- Collapsible filter panel with 7 filter types
- Multi-select for categories and priorities
- Date range picker and priority score filtering

### Email Quick Actions
- Hover-reveal buttons (star, archive, delete)
- Color-coded action buttons with smooth transitions
- Confirmation dialogs for destructive actions

### Analytics Dashboard
- 10+ interactive charts with Recharts
- Time-series analysis with trend indicators
- Top senders table with volume metrics
- Auto-refreshing real-time data

## 🎯 Key Improvements in Latest Version

### Design Overhaul
✅ Replaced purple gradient colors with professional blue/gray palette  
✅ Removed card-based analytics, implemented clean borderless design  
✅ Added bottom accent bars (3px) on hover instead of full borders  
✅ GitHub-style underline tabs replacing pill-style tabs  
✅ Minimal shadows for subtle depth  

### New Features
✅ Advanced search with 7 filter parameters  
✅ Email quick actions (star, archive, delete) with hover reveal  
✅ Bulk operations (mark read/unread, delete multiple)  
✅ ML-based importance scoring with 6 factors  
✅ Unread dot indicator with pulse animation  
✅ Trend indicators (📈/📉) with color-coded percentages  

### Performance & UX
✅ Optimized console logging (SQL queries disabled)  
✅ Auto-refresh analytics every 60 seconds  
✅ Smooth transitions and animations (0.2s ease)  
✅ Responsive design for mobile and desktop  
✅ Error handling and user feedback  

### 3. Initialize the Application

1. Open your browser to `http://localhost:3000`
2. Click the **"Generate Emails"** button to:
   - Generate 50 AI-created professional emails
   - Train the ML classifier
   - Populate the database
   - Calculate statistics

## 📊 Project Structure

```
MailMate/
├── backend/
│   ├── app/
│   │   ├── models/          # Database models and schemas
│   │   │   ├── email.py     # Email SQLAlchemy model
│   │   │   └── schemas.py   # Pydantic schemas
│   │   ├── routers/         # API endpoints
│   │   │   ├── emails.py    # Email CRUD + advanced search + bulk ops + importance scoring
│   │   │   ├── analytics.py # Advanced analytics with time-series data
│   │   │   └── digest.py    # Daily digest & TTS
│   │   ├── services/        # Business logic
│   │   │   ├── email_generator.py  # AI email generation
│   │   │   └── data_cleaner.py     # Data preprocessing
│   │   ├── ml/              # Machine learning
│   │   │   └── classifier.py       # ML classification
│   │   ├── config.py        # Configuration
│   │   └── database.py      # Database connection
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Environment template
│
└── frontend/
    ├── src/
    │   ├── components/      # React components
    │   │   ├── EmailList.jsx       # Email list with quick actions
    │   │   ├── Statistics.jsx      # Analytics dashboard (redesigned)
    │   │   ├── AdvancedSearch.jsx  # NEW: Advanced search with filters
    │   │   ├── DailyDigest.jsx     # Daily digest with TTS
    │   │   └── Filters.jsx         # Basic filters
    │   ├── services/        # API integration
    │   │   └── api.js
    │   ├── styles/          # CSS styles
    │   │   ├── App.css             # Global styles with new color scheme
    │   │   ├── Statistics.css      # Clean professional analytics styles
    │   │   └── AdvancedSearch.css  # NEW: Search component styles
    │   ├── App.jsx          # Main application with feature handlers
    │   └── main.jsx         # Entry point
    ├── package.json         # Node dependencies
    ├── vite.config.js       # Vite configuration
    └── index.html           # HTML template
```

## 🔧 API Endpoints

### Email Management
- `POST /api/emails/generate` - Generate 50 AI emails and train ML model
- `GET /api/emails/` - Get all emails (with filters)
- `GET /api/emails/search` - **NEW** Advanced search with 7 filter parameters
- `GET /api/emails/{id}` - Get specific email
- `GET /api/emails/{id}/importance` - **NEW** Get ML-based importance score (0-100) with recommendations
- `PUT /api/emails/{id}` - Update email (mark as read, etc.)
- `DELETE /api/emails/{id}` - Delete email
- `GET /api/emails/stats/summary` - Get email statistics
- `GET /api/emails/priority/top` - Get top priority emails
- `POST /api/emails/retrain` - Retrain ML classifier

### Bulk Operations (NEW)
- `POST /api/emails/bulk/mark-read` - Mark multiple emails as read
- `POST /api/emails/bulk/mark-unread` - Mark multiple emails as unread
- `DELETE /api/emails/bulk/delete` - Delete multiple emails in one operation

### Analytics (ENHANCED)
- `GET /api/analytics/advanced` - Comprehensive analytics with time-series data, volume trends, and sender statistics

### Daily Digest
- `GET /api/digest/daily` - Get daily digest (top 5 emails)
- `POST /api/digest/speak` - Generate speech from text
- `GET /api/digest/speak-digest` - Generate speech for daily digest

## 🧠 Machine Learning Details

### Classification Model
- **Algorithm**: Multinomial Naive Bayes (can switch to Logistic Regression)
- **Features**: TF-IDF vectorization with 1000 max features
- **Categories**: Work, Personal, Urgent, Promotion
- **Priorities**: High, Medium, Low

### Data Cleaning Pipeline
1. Text normalization (lowercase)
2. Remove email addresses and URLs
3. Remove special characters
4. Stop word removal
5. Feature extraction (urgency, promotional indicators)

### Priority Scoring
- Based on ML model confidence scores
- Considers category (urgent = high priority)
- Uses TF-IDF weighted features

## 🎨 Features in Detail

### 1. Advanced Search (NEW)
- **Full-text search** across subject, sender, and body
- **Multi-select filters** for categories (Work, Personal, Urgent, Promotion, Other)
- **Priority filtering** with multi-select (High, Medium, Low)
- **Read status filter** (All, Unread, Read)
- **Date range picker** for filtering by time period
- **Min priority score** filter (0.0-1.0 threshold)
- **Collapsible filter panel** for clean UI when not in use
- Query parameter structure: `?q=text&categories[]=WORK&priorities[]=HIGH&is_read=false&start_date=2024-01-01&end_date=2024-12-31&min_priority_score=0.7`

### 2. Email Quick Actions (NEW)
- **Star/Unstar**: Click star icon to mark important emails (yellow indicator)
- **Archive**: Remove emails from inbox view with archive button (blue hover)
- **Delete**: Permanently delete with confirmation dialog (red hover)
- **Hover reveal**: Action buttons appear smoothly on email hover
- **Visual feedback**: Icons from React Icons library with color-coded states
- **Unread indicator**: Pulsing blue dot for unread emails

### 3. Bulk Operations (NEW)
- **Mark all as read**: Process multiple emails at once
- **Mark all as unread**: Batch unread operation
- **Bulk delete**: Delete multiple emails with single API call
- **Efficient processing**: Returns count of affected emails
- **Request format**: `{ "email_ids": [1, 2, 3, 4, 5] }`

### 4. Importance Scoring (NEW)
- **ML-based algorithm** with 6 scoring factors:
  - **Recency score** (0-10): Time-based freshness calculation
  - **Read status** (0/15): Unread emails get bonus points
  - **Sender frequency** (0-20): How often sender communicates
  - **Content length** (0-15): Longer emails weighted higher
  - **Attachments** (0/10): Bonus for emails with attachments
  - **Category urgency** (0-30): URGENT category gets max points
- **Score range**: 0-100 with confidence percentage
- **Recommendations**: CRITICAL (90+), HIGH (70-89), MEDIUM (50-69), LOW (30-49), MINIMAL (<30)
- **Response format**: `{ "email_id": 1, "importance_score": 85, "factors": {...}, "recommendation": "HIGH" }`

### 5. Email Generation
- Uses Faker library for realistic names and data
- Template-based generation for different categories
- Realistic timestamps (within last 30 days)
- Professional email formats
- Automatically generates 50 emails on first run

### 6. Data Visualization (ENHANCED)
- **10+ interactive charts** using Recharts
- **Bar Charts**: Category distribution, sender volume
- **Line Charts**: Email volume over time with trend lines
- **Area Charts**: Read vs unread trends over time
- **Pie Charts**: Priority distribution, read/unread ratio
- **Time-series analysis**: Hourly patterns, daily trends
- **Top senders table**: Most frequent email senders with volume metrics
- **Responsive tooltips**: Hover for detailed information

### 7. Professional UI Design (REDESIGNED)
- **GitHub-inspired color scheme**:
  - Primary: #0366d6 (blue), #044289 (dark blue)
  - Secondary: #586069 (gray), #24292e (near-black)
  - Background: #f6f8fa (light gray), #ffffff (white)
  - Borders: #e1e4e8 (subtle gray)
- **Borderless stat cards** with bottom accent bars (3px on hover)
- **Clean tables** with gray headers (#f6f8fa) instead of gradients
- **GitHub-style tabs** with underline active state
- **Minimal shadows** for subtle depth
- **Professional buttons** with clear hover states
- **Removed**: Purple gradients, heavy card shadows, unprofessional colors

### 8. Real-time Analytics (ENHANCED)
- **Summary statistics** with trend indicators (📈/📉)
- **Color-coded trends**: Green for positive, red for negative
- **Volume change percentage** with comparison to previous period
- **Auto-refresh**: Updates every 60 seconds
- **Tab navigation**: Overview, Time Analysis, Top Senders
- **Statistical insights**: Average priority, read rates, category distribution

### 9. Daily Digest with TTS
- Automatically selects top 5 priority emails
- Generates summary text
- Uses gTTS (Google Text-to-Speech) backend
- Web Speech API fallback in browser
- Read subject, sender, and email summary

### 10. Search & Filter
- Real-time search across subject, sender, and body
- Advanced filter panel with multiple criteria
- Combinable filters for precise results
- Clear filters button to reset all selections

## 🛠️ Technologies Used

### Backend
- **FastAPI 0.115**: Modern, fast web framework
- **SQLAlchemy 2.0**: ORM for database operations
- **PyMySQL**: MySQL database driver
- **Pandas 2.3**: Data manipulation and analysis
- **NumPy 2.3**: Numerical computing
- **Scikit-learn 1.7**: Machine learning library
- **Faker**: Fake data generation
- **gTTS**: Google Text-to-Speech
- **Uvicorn 0.30**: ASGI server

### Frontend
- **React 18**: UI framework with hooks
- **Vite 5**: Build tool and dev server
- **Recharts 2.10**: Charting library for data visualization
- **Axios**: HTTP client for API calls
- **React Icons**: Icon library (FaStar, FaArchive, FaTrash, FaSearch, etc.)
- **date-fns**: Date formatting and manipulation

### Database
- **MySQL 8.0**: Relational database
- **SQLAlchemy ORM**: Python SQL toolkit

## 📝 Usage Guide

### Generate Emails
1. Click "Generate Emails" button
2. System generates 50 professional emails
3. ML model trains on the data
4. Emails are classified and prioritized automatically
5. Statistics automatically calculated

### Advanced Search (NEW)
1. Click search bar at top of inbox
2. Enter search query for full-text search
3. Click filter icon to expand advanced filters
4. Select multiple categories (checkboxes)
5. Select multiple priorities (checkboxes)
6. Choose read status (dropdown)
7. Set date range (start and end dates)
8. Set minimum priority score (0.0-1.0)
9. Click "Apply Filters" to search
10. Click "Clear Filters" to reset

### Email Quick Actions (NEW)
- **Star email**: Hover over email, click star icon (toggles yellow)
- **Archive email**: Hover, click archive icon (removes from view)
- **Delete email**: Hover, click trash icon, confirm deletion
- **Mark as read**: Click on email row
- **View importance**: Open email detail to see importance score

### Bulk Operations (NEW)
- Select multiple emails (checkbox implementation pending)
- Use bulk actions in toolbar:
  - Mark all selected as read
  - Mark all selected as unread
  - Delete all selected (with confirmation)

### Browse Emails
- Scroll through email list
- Unread emails show pulsing blue dot
- Click email to mark as read
- View category and priority badges
- Hover to reveal quick action buttons

### Filter Emails
- Use advanced search for complex queries
- Search box for quick keyword search
- Select category from filter panel
- Select priority level from filter panel
- Combine multiple filters

### View Statistics & Analytics
- **Summary cards** at top: Total, Unread, Avg Priority, Volume Trend
- **Trend indicators**: 📈 for increase, 📉 for decrease
- **Color-coded changes**: Green positive, red negative
- **Tab navigation**: Switch between Overview, Time Analysis, Top Senders
- **Interactive charts**: Hover for detailed tooltips
- **Auto-refresh**: Data updates every 60 seconds

### Check Email Importance (NEW)
1. Select an email
2. View importance score (0-100)
3. See breakdown of 6 scoring factors
4. Read recommendation (CRITICAL/HIGH/MEDIUM/LOW/MINIMAL)
5. Use scores to prioritize reading order

### Daily Digest
- View top 5 priority emails
- Click "Read Aloud" to hear digest
- Uses Text-to-Speech to read summaries
- Click "Stop" to stop playback

### Retrain Model
- Click "Retrain Model" after adding more emails
- Improves classification accuracy
- Updates priority scores

## 🔍 Troubleshooting

### Database Connection Issues
```powershell
# Check MySQL is running
Get-Service MySQL*

# Start MySQL if stopped
Start-Service MySQL80
```

### Backend Issues
```powershell
# Check Python version
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend Issues
```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
Remove-Item -Recurse -Force node_modules
npm install
```

### CORS Issues
- Ensure backend CORS settings include frontend URL
- Check both servers are running on correct ports

## 🚀 Advanced Features

### Advanced Search System
- **7 filter parameters** for precise email discovery
- **Query string building** with URLSearchParams
- **Multi-select support** for categories and priorities
- **Date range filtering** for time-based searches
- **Priority score threshold** for quality filtering
- **Collapsible UI** to save screen space

### Email Importance Algorithm
The importance scoring system uses a weighted multi-factor approach:

```python
# Scoring Formula (0-100 scale)
importance_score = (
    recency_score +          # 0-10 points (time-based decay)
    read_status_score +      # 0 or 15 points (unread bonus)
    sender_frequency_score + # 0-20 points (frequent sender = important)
    content_length_score +   # 0-15 points (length proxy for detail)
    attachment_score +       # 0 or 10 points (attachment bonus)
    category_urgency_score   # 0-30 points (URGENT category max)
)
```

**Recency Calculation**: Uses 10-point decay over 30 days
**Sender Frequency**: Analyzes historical email volume from sender
**Recommendations**: 
- CRITICAL: 90-100 (immediate attention required)
- HIGH: 70-89 (prioritize soon)
- MEDIUM: 50-69 (review when convenient)
- LOW: 30-49 (low priority)
- MINIMAL: 0-29 (can be deferred)











---

**Note**: This is a simulated email system for demonstration purposes. For production use, implement proper security, authentication, and use real email APIs.




