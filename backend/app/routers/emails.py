from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, func
from typing import List, Optional
from app.database import get_db
from app.models.email import Email, EmailCategory, EmailPriority
from app.models.schemas import EmailResponse, EmailCreate, EmailUpdate
from app.services.email_generator import generate_50_emails
from app.services.data_cleaner import EmailDataCleaner
from app.ml.classifier import EmailClassifier
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

router = APIRouter(prefix="/api/emails", tags=["emails"])

# Global instances
data_cleaner = EmailDataCleaner()
classifier = EmailClassifier(model_type="naive_bayes")

@router.post("/generate", response_model=dict)
async def generate_emails(db: Session = Depends(get_db)):
    """Generate 50 AI-created professional emails and store in database"""
    try:
        # Generate emails
        emails = generate_50_emails()
        
        # Clean the data
        emails_df = data_cleaner.preprocess_emails(emails)
        
        # Train the classifier on generated emails
        training_metrics = classifier.train(emails_df)
        
        # Predict categories and priorities
        emails_df = classifier.predict_batch(emails_df)
        
        # Store in database
        db_emails = []
        for idx, row in emails_df.iterrows():
            email = Email(
                subject=row['subject'],
                sender=row['sender'],
                sender_email=row['sender_email'],
                body=row['body'],
                timestamp=row['timestamp'],
                category=EmailCategory(row['predicted_category']),
                priority=EmailPriority(row['predicted_priority']),
                priority_score=float(row['priority_score'])
            )
            db.add(email)
            db_emails.append(email)
        
        db.commit()
        
        # Get statistics
        stats = data_cleaner.get_statistics(emails_df)
        
        return {
            "message": "Successfully generated and stored 50 emails",
            "count": len(db_emails),
            "training_metrics": training_metrics,
            "statistics": stats
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[EmailResponse])
async def get_emails(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = None,
    priority: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get emails with optional filtering"""
    query = db.query(Email)
    
    # Apply filters
    if category:
        query = query.filter(Email.category == category)
    
    if priority:
        query = query.filter(Email.priority == priority)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Email.subject.ilike(search_term),
                Email.sender.ilike(search_term),
                Email.body.ilike(search_term)
            )
        )
    
    # Order by timestamp descending (newest first)
    query = query.order_by(desc(Email.timestamp))
    
    emails = query.offset(skip).limit(limit).all()
    return emails

@router.get("/{email_id}", response_model=EmailResponse)
async def get_email(email_id: int, db: Session = Depends(get_db)):
    """Get a specific email by ID"""
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    return email

@router.put("/{email_id}", response_model=EmailResponse)
async def update_email(
    email_id: int,
    email_update: EmailUpdate,
    db: Session = Depends(get_db)
):
    """Update email properties (mark as read, change category, etc.)"""
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Update fields
    if email_update.is_read is not None:
        email.is_read = email_update.is_read
    
    if email_update.category is not None:
        email.category = EmailCategory(email_update.category)
    
    if email_update.priority is not None:
        email.priority = EmailPriority(email_update.priority)
    
    db.commit()
    db.refresh(email)
    
    return email

@router.delete("/{email_id}")
async def delete_email(email_id: int, db: Session = Depends(get_db)):
    """Delete an email"""
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    db.delete(email)
    db.commit()
    
    return {"message": "Email deleted successfully"}

@router.get("/stats/summary")
async def get_email_statistics(db: Session = Depends(get_db)):
    """Get email statistics and analytics"""
    emails = db.query(Email).all()
    
    if not emails:
        return {
            "total_emails": 0,
            "category_distribution": {},
            "priority_distribution": {},
            "read_unread": {"read": 0, "unread": 0}
        }
    
    # Convert to DataFrame for analysis
    email_dicts = [email.to_dict() for email in emails]
    df = pd.DataFrame(email_dicts)
    
    # Calculate statistics
    stats = {
        "total_emails": len(emails),
        "category_distribution": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
        "priority_distribution": df['priority'].value_counts().to_dict() if 'priority' in df.columns else {},
        "read_unread": {
            "read": int(df['is_read'].sum()) if 'is_read' in df.columns else 0,
            "unread": int((df['is_read'] == 0).sum()) if 'is_read' in df.columns else 0
        },
        "avg_priority_score": float(df['priority_score'].mean()) if 'priority_score' in df.columns else 0
    }
    
    return stats

@router.get("/priority/top")
async def get_top_priority_emails(
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Get top priority emails for daily digest"""
    emails = db.query(Email).order_by(
        desc(Email.priority_score),
        desc(Email.timestamp)
    ).limit(limit).all()
    
    return [email.to_dict() for email in emails]

@router.post("/retrain")
async def retrain_classifier(db: Session = Depends(get_db)):
    """Retrain the ML classifier with current database emails"""
    try:
        emails = db.query(Email).all()
        
        if not emails:
            raise HTTPException(status_code=400, detail="No emails available for training")
        
        # Convert to DataFrame
        email_dicts = [email.to_dict() for email in emails]
        df = pd.DataFrame(email_dicts)
        
        # Clean the data
        df_cleaned = data_cleaner.preprocess_emails(email_dicts)
        
        # Retrain classifier
        training_metrics = classifier.train(df_cleaned)
        
        return {
            "message": "Classifier retrained successfully",
            "metrics": training_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Search and Filtering
@router.get("/search", response_model=List[EmailResponse])
async def search_emails(
    q: Optional[str] = Query(None, description="Search query for subject, sender, or body"),
    categories: Optional[List[str]] = Query(None, description="Filter by categories"),
    priorities: Optional[List[str]] = Query(None, description="Filter by priorities"),
    is_read: Optional[bool] = Query(None, description="Filter by read status"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    min_priority_score: Optional[float] = Query(None, description="Minimum priority score"),
    limit: int = Query(100, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Advanced search with multiple filters"""
    try:
        query = db.query(Email)
        
        # Text search across multiple fields
        if q:
            search_filter = or_(
                Email.subject.ilike(f"%{q}%"),
                Email.sender.ilike(f"%{q}%"),
                Email.body.ilike(f"%{q}%")
            )
            query = query.filter(search_filter)
        
        # Category filter
        if categories:
            query = query.filter(Email.category.in_(categories))
        
        # Priority filter
        if priorities:
            query = query.filter(Email.priority.in_(priorities))
        
        # Read status filter
        if is_read is not None:
            query = query.filter(Email.is_read == (1 if is_read else 0))
        
        # Date range filter
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(Email.received_at >= start_dt)
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            query = query.filter(Email.received_at <= end_dt)
        
        # Priority score filter
        if min_priority_score is not None:
            query = query.filter(Email.priority_score >= min_priority_score)
        
        # Order by most recent
        query = query.order_by(desc(Email.received_at))
        
        # Limit results
        emails = query.limit(limit).all()
        
        return emails
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bulk Operations
@router.post("/bulk/mark-read", response_model=dict)
async def bulk_mark_read(
    email_ids: List[int] = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Mark multiple emails as read"""
    try:
        updated = db.query(Email).filter(Email.id.in_(email_ids)).update(
            {"is_read": 1}, synchronize_session=False
        )
        db.commit()
        return {"message": f"Marked {updated} emails as read", "count": updated}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk/mark-unread", response_model=dict)
async def bulk_mark_unread(
    email_ids: List[int] = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Mark multiple emails as unread"""
    try:
        updated = db.query(Email).filter(Email.id.in_(email_ids)).update(
            {"is_read": 0}, synchronize_session=False
        )
        db.commit()
        return {"message": f"Marked {updated} emails as unread", "count": updated}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/bulk/delete", response_model=dict)
async def bulk_delete_emails(
    email_ids: List[int] = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Delete multiple emails"""
    try:
        deleted = db.query(Email).filter(Email.id.in_(email_ids)).delete(synchronize_session=False)
        db.commit()
        return {"message": f"Deleted {deleted} emails", "count": deleted}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Email Importance Scoring
@router.get("/{email_id}/importance", response_model=dict)
async def get_email_importance(
    email_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed importance score and factors for an email"""
    try:
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail="Email not found")
        
        # Calculate importance factors
        factors = {
            "priority_score": email.priority_score,
            "is_high_priority": email.priority == "HIGH",
            "is_urgent_category": email.category == "URGENT",
            "is_unread": email.is_read == 0,
            "recency_score": calculate_recency_score(email.received_at),
            "sender_frequency": get_sender_frequency(db, email.sender),
        }
        
        # Calculate overall importance (0-100)
        importance = (
            factors["priority_score"] * 30 +
            (50 if factors["is_high_priority"] else 0) +
            (30 if factors["is_urgent_category"] else 0) +
            (20 if factors["is_unread"] else 0) +
            factors["recency_score"] * 10 +
            min(factors["sender_frequency"] * 2, 20)
        ) / 2
        
        return {
            "email_id": email_id,
            "importance_score": round(importance, 2),
            "factors": factors,
            "recommendation": get_importance_recommendation(importance)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_recency_score(received_at: datetime) -> float:
    """Calculate score based on how recent the email is (0-10)"""
    hours_ago = (datetime.now() - received_at).total_seconds() / 3600
    if hours_ago < 1:
        return 10.0
    elif hours_ago < 24:
        return 8.0
    elif hours_ago < 72:
        return 5.0
    elif hours_ago < 168:  # 1 week
        return 3.0
    else:
        return 1.0

def get_sender_frequency(db: Session, sender: str) -> int:
    """Get number of emails from this sender"""
    return db.query(func.count(Email.id)).filter(Email.sender == sender).scalar()

def get_importance_recommendation(score: float) -> str:
    """Get recommendation based on importance score"""
    if score >= 80:
        return "CRITICAL - Respond immediately"
    elif score >= 60:
        return "HIGH - Respond today"
    elif score >= 40:
        return "MEDIUM - Respond within 2-3 days"
    elif score >= 20:
        return "LOW - Respond when convenient"
    else:
        return "MINIMAL - Can be deferred or archived"

