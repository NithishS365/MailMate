from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class EmailBase(BaseModel):
    subject: str
    sender: str
    sender_email: EmailStr
    body: str
    
class EmailCreate(EmailBase):
    timestamp: Optional[datetime] = None
    
class EmailResponse(EmailBase):
    id: int
    timestamp: datetime
    category: Optional[str] = None
    priority: Optional[str] = None
    priority_score: float
    is_read: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class EmailUpdate(BaseModel):
    is_read: Optional[int] = None
    category: Optional[str] = None
    priority: Optional[str] = None
