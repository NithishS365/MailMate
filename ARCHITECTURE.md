# MailMate - Project Overview

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MAILMATE SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐  │
│  │   Frontend   │ ───▶ │   Backend    │ ───▶│   MySQL   │  │
│  │    React     │      │   FastAPI    │      │ Database  │  │
│  │   (Port 3000)│ ◀─── │  (Port 8000) │ ◀───│           │  │
│  └──────────────┘      └──────────────┘      └───────────┘  │
│         │                      │                            │
│         │                      │                            │
│    ┌────▼────┐           ┌────▼─────┐                       │
│    │ Recharts│           │ ML Model │                       │
│    │  Charts │           │  (Naive  │                       │
│    │   TTS   │           │  Bayes)  │                       │
│    └─────────┘           └──────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Email Generation Flow
```
User clicks "Generate Emails"
    ↓
Frontend sends POST /api/emails/generate
    ↓
Backend generates 50 AI emails (Faker)
    ↓
Data cleaning with Pandas
    ↓
ML model training (TF-IDF + Naive Bayes)
    ↓
Classification & Priority scoring
    ↓
Store in MySQL database
    ↓
Return statistics to frontend
```

### 2. Email Display Flow
```
Frontend requests GET /api/emails/
    ↓
Backend queries MySQL
    ↓
Apply filters (category, priority, search)
    ↓
Return JSON array
    ↓
Frontend displays in EmailList component
    ↓
Recharts visualizes distribution
```

### 3. Daily Digest Flow
```
User navigates to Daily Digest
    ↓
Frontend requests GET /api/digest/daily
    ↓
Backend queries top 5 priority emails
    ↓
Generate summary text
    ↓
User clicks "Read Aloud"
    ↓
Generate speech with gTTS
    ↓
Browser plays with Web Speech API
```

## Technology Stack

### Backend Technologies
- **FastAPI**: High-performance web framework
- **SQLAlchemy**: ORM for database operations
- **Pandas**: Data manipulation and cleaning
- **Scikit-learn**: Machine learning (Naive Bayes)
- **Faker**: Generate realistic email data
- **gTTS**: Text-to-speech conversion
- **PyMySQL**: MySQL connector

### Frontend Technologies
- **React 18**: Component-based UI
- **Vite**: Fast build tool
- **Recharts**: Data visualization
- **Axios**: HTTP requests
- **React Icons**: Icon components
- **date-fns**: Date formatting

### Data Science Components
- **TF-IDF Vectorization**: Convert text to features
- **Naive Bayes Classifier**: Email categorization
- **Priority Scoring**: Confidence-based ranking
- **Data Cleaning Pipeline**: Text preprocessing
- **Feature Engineering**: Extract email indicators

## Database Schema

```sql
Table: emails
├── id (INT, PRIMARY KEY)
├── subject (VARCHAR(500))
├── sender (VARCHAR(200))
├── sender_email (VARCHAR(200))
├── body (TEXT)
├── timestamp (DATETIME)
├── category (ENUM: work, personal, urgent, promotion)
├── priority (ENUM: high, medium, low)
├── priority_score (FLOAT)
├── is_read (INT: 0 or 1)
├── created_at (DATETIME)
└── updated_at (DATETIME)
```

## ML Model Details

### Training Process
1. **Data Collection**: 50 generated emails with labels
2. **Preprocessing**: Clean text, remove stop words
3. **Vectorization**: TF-IDF with 1000 features
4. **Training**: Naive Bayes on 80% data
5. **Validation**: Test on 20% data
6. **Prediction**: Classify new emails

### Features Used
- **Text Features**: TF-IDF from subject + body
- **Category Indicators**: Urgency, promotional, work keywords
- **Priority Mapping**: Category → Priority level

### Model Performance
- Typical accuracy: 85-95% (depends on data)
- Fast inference: < 10ms per email
- Retrainable: Can update with new data

## API Endpoints Reference

### Email Endpoints
| Method | Endpoint | Description  |
|--------|----------|------------- |
| POST | `/api/emails/generate` | Generate 50 emails & train ML |
| GET | `/api/emails/` | Get all emails (filterable) |
| GET | `/api/emails/{id}` | Get single email |
| PUT | `/api/emails/{id}` | Update email |
| DELETE | `/api/emails/{id}` | Delete email |
| GET | `/api/emails/stats/summary` | Get statistics |
| GET | `/api/emails/priority/top` | Get top N priority emails |
| POST | `/api/emails/retrain` | Retrain ML model |

### Digest Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/digest/daily` | Get daily digest |
| POST | `/api/digest/speak` | Generate TTS audio |
| GET | `/api/digest/speak-digest` | TTS for digest |

## Component Hierarchy

```
App.jsx
├── Header
│   ├── Title
│   └── Actions (Generate, Retrain buttons)
├── Messages (Success/Error)
└── Dashboard
    ├── EmailSection
    │   ├── Filters
    │   │   ├── SearchInput
    │   │   ├── CategorySelect
    │   │   └── PrioritySelect
    │   └── EmailList
    │       └── EmailItem[] (map)
    └── Sidebar
        ├── Statistics
        │   ├── StatCards
        │   ├── CategoryChart (Bar)
        │   ├── PriorityChart (Pie)
        │   └── ReadUnreadChart (Pie)
        └── DailyDigest
            ├── DigestHeader
            ├── DigestEmails[]
            └── TTSControls
```

## State Management

### App-level State
- `emails`: All emails from database
- `filteredEmails`: Filtered subset
- `statistics`: Aggregate stats
- `loading`: Loading indicator
- `message`: Success messages
- `error`: Error messages
- `activeFilters`: Current filter values

### Component State
- **Filters**: Search text, category, priority
- **DailyDigest**: Digest data, speaking status
- **Statistics**: Chart data transformations

## Security Considerations

### Current Implementation (Demo)
- ⚠️ No authentication
- ⚠️ No rate limiting
- ⚠️ No input sanitization
- ⚠️ Passwords in plain .env

### Production Requirements
- ✅ Add JWT authentication
- ✅ Implement rate limiting
- ✅ Sanitize all inputs
- ✅ Use environment variables securely
- ✅ Add HTTPS
- ✅ Implement RBAC
- ✅ Add request validation
- ✅ SQL injection prevention (SQLAlchemy handles this)

## Performance Optimization

### Backend
- Database indexing on frequently queried fields
- Connection pooling (SQLAlchemy)
- Async operations where possible
- Caching for statistics

### Frontend
- Lazy loading for large lists
- Debounced search
- Memoization for expensive computations
- Code splitting

### ML Model
- Pre-trained model caching
- Batch prediction for multiple emails
- TF-IDF vectorizer reuse

## Deployment Guide

### Backend Deployment
1. Use Gunicorn/Uvicorn workers
2. Set up reverse proxy (Nginx)
3. Configure production database
4. Set environment variables
5. Enable HTTPS

### Frontend Deployment
1. Build production bundle: `npm run build`
2. Serve with Nginx/Apache
3. Configure API endpoint
4. Enable CDN

### Database
1. Production MySQL instance
2. Automated backups
3. Read replicas for scaling
4. Query optimization

## Testing Strategy

### Backend Tests
- Unit tests for services
- Integration tests for APIs
- ML model validation tests
- Database connection tests

### Frontend Tests
- Component unit tests
- Integration tests
- E2E tests with Playwright
- Accessibility tests

### ML Tests
- Model accuracy tests
- Feature engineering validation
- Classification correctness
- Edge case handling

## Future Roadmap

### Phase 1 (Current)
- ✅ Email generation
- ✅ ML classification
- ✅ Basic dashboard
- ✅ Daily digest with TTS

### Phase 2 (Next)
- ⬜ User authentication
- ⬜ Real Gmail API integration
- ⬜ Email threads
- ⬜ Advanced search

### Phase 3 (Future)
- ⬜ Sentiment analysis
- ⬜ Auto-reply suggestions
- ⬜ Email scheduling
- ⬜ Mobile app

### Phase 4 (Advanced)
- ⬜ Multi-account support
- ⬜ Team collaboration
- ⬜ Analytics dashboard
- ⬜ API for third-party integration

## License & Credits

**Created for educational purposes**

### Technologies Used
- FastAPI (MIT License)
- React (MIT License)
- Scikit-learn (BSD License)
- Recharts (MIT License)
- And many other open-source libraries

### Acknowledgments
Thank you to all open-source contributors who made this project possible!

---

**MailMate** - Demonstrating the power of Python, Machine Learning, and Modern Web Development! 🚀
