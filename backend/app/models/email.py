from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Enum
from datetime import datetime
from app.database import Base
import enum

class EmailCategory(str, enum.Enum):
    WORK = "work"
    PERSONAL = "personal"
    URGENT = "urgent"
    PROMOTION = "promotion"
    SPAM = "spam"

class EmailPriority(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Email(Base):
    __tablename__ = "emails"
    
    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(500), nullable=False)
    sender = Column(String(200), nullable=False)
    sender_email = Column(String(200), nullable=False)
    body = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    category = Column(Enum(EmailCategory), nullable=True)
    priority = Column(Enum(EmailPriority), default=EmailPriority.MEDIUM)
    priority_score = Column(Float, default=0.5)
    is_read = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject,
            "sender": self.sender,
            "sender_email": self.sender_email,
            "body": self.body,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "category": self.category.value if self.category else None,
            "priority": self.priority.value if self.priority else None,
            "priority_score": self.priority_score,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
