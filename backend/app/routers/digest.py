from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.database import get_db
from app.models.email import Email
from gtts import gTTS
import os
import tempfile
from typing import List

router = APIRouter(prefix="/api/digest", tags=["digest"])

@router.get("/daily")
async def get_daily_digest(db: Session = Depends(get_db)):
    """Get daily digest: top 5 most important emails"""
    # Get top 5 priority emails
    emails = db.query(Email).order_by(
        desc(Email.priority_score),
        desc(Email.timestamp)
    ).limit(5).all()
    
    if not emails:
        return {
            "message": "No emails available for digest",
            "emails": [],
            "summary": ""
        }
    
    # Create summary
    summary_parts = []
    for idx, email in enumerate(emails, 1):
        # Create a short summary (first 100 chars of body)
        body_summary = email.body[:100] + "..." if len(email.body) > 100 else email.body
        
        summary_parts.append(
            f"Email {idx}: From {email.sender}. Subject: {email.subject}. "
            f"Priority: {email.priority.value if email.priority else 'medium'}. "
            f"Summary: {body_summary}"
        )
    
    full_summary = " Next email. ".join(summary_parts)
    
    return {
        "message": "Daily digest generated",
        "emails": [email.to_dict() for email in emails],
        "summary": full_summary,
        "count": len(emails)
    }

@router.post("/speak")
async def generate_speech(text: str):
    """Generate speech from text using Text-to-Speech"""
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        
        return {
            "message": "Speech generated successfully",
            "audio_path": temp_path,
            "text_length": len(text)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@router.get("/speak-digest")
async def speak_daily_digest(db: Session = Depends(get_db)):
    """Generate speech for daily digest"""
    # Get digest
    digest_data = await get_daily_digest(db)
    
    if not digest_data["emails"]:
        return {
            "message": "No emails to speak",
            "audio_generated": False
        }
    
    # Generate speech intro
    intro = f"Good day! Here is your daily email digest with {digest_data['count']} important emails. "
    full_text = intro + digest_data['summary']
    
    # Generate speech
    speech_result = await generate_speech(full_text)
    
    return {
        "message": "Daily digest speech generated",
        "audio_path": speech_result["audio_path"],
        "emails_count": digest_data['count'],
        "text_length": len(full_text)
    }
