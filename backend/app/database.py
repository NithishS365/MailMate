from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Disable SQL logging for cleaner output
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables and generate sample data ONCE if empty"""
    Base.metadata.create_all(bind=engine)
    
    # Check if emails table is empty
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT COUNT(*) FROM emails"))
        count = result.scalar()
        
        if count == 0:
            # Generate sample emails ONLY if database is completely empty
            import pandas as pd
            from app.services.email_generator import generate_50_emails
            from app.ml.classifier import EmailClassifier
            
            print("\n" + "="*60)
            print("🔄 INITIALIZING MAILMATE DATABASE (ONE-TIME SETUP)")
            print("="*60)
            print("📧 Generating 50 AI-created professional emails...")
            emails_data = generate_50_emails()
            print(f"✅ Generated {len(emails_data)} emails successfully!")
            
            # Convert list of dicts to DataFrame for ML training
            emails_df = pd.DataFrame(emails_data)
            
            print("\n🧠 Training Machine Learning classifier...")
            classifier = EmailClassifier()
            classifier.train(emails_df)
            print("✅ ML model trained successfully!")
            
            print("\n💾 Saving emails to database with classifications...")
            # Save emails to database with ML predictions
            for idx, email_data in enumerate(emails_data, 1):
                # Get predictions from classifier
                email_text = email_data['subject'] + ' ' + email_data['body']
                category, _ = classifier.predict_category(email_text)
                priority, priority_score = classifier.predict_priority(email_text)
                
                # Add ML predictions to email data
                email_data['category'] = category
                email_data['priority'] = priority
                email_data['priority_score'] = float(priority_score)
                
                db.execute(
                    text("""
                        INSERT INTO emails (subject, sender, sender_email, body, timestamp, 
                                          category, priority, priority_score, is_read, 
                                          created_at, updated_at)
                        VALUES (:subject, :sender, :sender_email, :body, :timestamp,
                                :category, :priority, :priority_score, :is_read,
                                NOW(), NOW())
                    """),
                    email_data
                )
                
                if idx % 10 == 0:
                    print(f"   📌 Processed {idx}/{len(emails_data)} emails...")
            
            db.commit()
            print(f"\n✅ ALL DONE! {len(emails_data)} emails saved and classified!")
            print("="*60)
            print("🚀 MailMate is ready to use!")
            print("   📊 {count} emails will be reused for all future requests")
            print("="*60 + "\n")
        else:
            print(f"\n✅ Database already contains {count} emails - using existing data.\n")
    except Exception as e:
        print(f"\n❌ Error initializing database: {e}\n")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
