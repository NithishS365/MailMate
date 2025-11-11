import pandas as pd
import re
from typing import List, Dict
import numpy as np

class EmailDataCleaner:
    """Data cleaning and preprocessing for emails using Pandas"""
    
    def __init__(self):
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        ])
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """Remove common stop words"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text"""
        cleaned = self.clean_text(text)
        cleaned = self.remove_stop_words(cleaned)
        
        words = cleaned.split()
        word_freq = pd.Series(words).value_counts()
        
        return word_freq.head(top_n).index.tolist()
    
    def preprocess_emails(self, emails: List[Dict]) -> pd.DataFrame:
        """
        Preprocess a list of emails and return cleaned DataFrame
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Cleaned pandas DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame(emails)
        
        # Handle missing values
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')
        df['sender'] = df['sender'].fillna('Unknown')
        
        # Create cleaned versions of text fields
        df['subject_cleaned'] = df['subject'].apply(self.clean_text)
        df['body_cleaned'] = df['body'].apply(self.clean_text)
        
        # Combine subject and body for full content analysis
        df['full_content'] = df['subject'] + ' ' + df['body']
        df['full_content_cleaned'] = df['full_content'].apply(self.clean_text)
        df['full_content_cleaned'] = df['full_content_cleaned'].apply(self.remove_stop_words)
        
        # Extract features
        df['subject_length'] = df['subject'].str.len()
        df['body_length'] = df['body'].str.len()
        df['word_count'] = df['body'].str.split().str.len()
        
        # Detect urgency indicators
        urgency_keywords = ['urgent', 'asap', 'immediate', 'critical', 'emergency', 'priority', 'action required']
        df['has_urgency'] = df['full_content'].str.lower().apply(
            lambda x: any(keyword in x for keyword in urgency_keywords)
        ).astype(int)
        
        # Detect promotional indicators
        promo_keywords = ['sale', 'discount', 'offer', 'deal', 'promotion', 'limited time', 'exclusive']
        df['has_promo'] = df['full_content'].str.lower().apply(
            lambda x: any(keyword in x for keyword in promo_keywords)
        ).astype(int)
        
        # Detect work-related indicators
        work_keywords = ['meeting', 'project', 'deadline', 'report', 'presentation', 'review', 'team', 'client']
        df['has_work'] = df['full_content'].str.lower().apply(
            lambda x: any(keyword in x for keyword in work_keywords)
        ).astype(int)
        
        # Calculate email age (days from now)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['email_age_days'] = (pd.Timestamp.now() - df['timestamp']).dt.days
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the email dataset"""
        stats = {
            "total_emails": len(df),
            "avg_subject_length": df['subject_length'].mean() if 'subject_length' in df.columns else 0,
            "avg_body_length": df['body_length'].mean() if 'body_length' in df.columns else 0,
            "avg_word_count": df['word_count'].mean() if 'word_count' in df.columns else 0,
            "urgent_count": df['has_urgency'].sum() if 'has_urgency' in df.columns else 0,
            "promo_count": df['has_promo'].sum() if 'has_promo' in df.columns else 0,
            "work_count": df['has_work'].sum() if 'has_work' in df.columns else 0,
        }
        
        if 'category' in df.columns:
            stats['category_distribution'] = df['category'].value_counts().to_dict()
        
        return stats
