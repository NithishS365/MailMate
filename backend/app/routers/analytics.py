from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.email import Email
import pandas as pd

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/advanced")
async def get_advanced_analytics(db: Session = Depends(get_db)):
    """Get comprehensive advanced analytics for visualization"""
    try:
        emails = db.query(Email).all()
        
        if not emails:
            return {
                "total_emails": 0,
                "analytics": {}
            }
        
        # Convert to DataFrame for advanced analysis
        email_dicts = [email.to_dict() for email in emails]
        df = pd.DataFrame(email_dicts)
        
        # 1. Category Distribution with Priority Breakdown
        category_priority = df.groupby(['category', 'priority']).size().reset_index(name='count')
        category_priority_data = []
        for category in df['category'].unique():
            cat_data = {'category': category.capitalize()}
            for priority in ['high', 'medium', 'low']:
                count = category_priority[
                    (category_priority['category'] == category) & 
                    (category_priority['priority'] == priority)
                ]['count'].sum()
                cat_data[priority] = int(count)
            category_priority_data.append(cat_data)
        
        # 2. Time-based Analysis (Emails over time)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = daily_counts['date'].astype(str)
        time_series_data = daily_counts.to_dict('records')
        
        # 3. Priority Score Distribution
        priority_bins = [0, 0.3, 0.6, 0.8, 1.0]
        priority_labels = ['Very Low', 'Low', 'Medium', 'High']
        df['priority_range'] = pd.cut(df['priority_score'], bins=priority_bins, labels=priority_labels)
        priority_distribution = df.groupby('priority_range', observed=True).size().reset_index(name='count')
        priority_distribution['priority_range'] = priority_distribution['priority_range'].astype(str)
        
        # 4. Sender Analysis (Top senders)
        sender_stats = df.groupby('sender').agg({
            'id': 'count',
            'is_read': lambda x: x.sum(),
            'priority_score': 'mean'
        }).reset_index()
        sender_stats.columns = ['sender', 'total_emails', 'read_emails', 'avg_priority']
        sender_stats = sender_stats.nlargest(10, 'total_emails')
        sender_stats['avg_priority'] = sender_stats['avg_priority'].round(3)
        
        # 5. Category + Read Status Matrix
        category_read = df.groupby(['category', 'is_read']).size().reset_index(name='count')
        category_read_matrix = []
        for category in df['category'].unique():
            read_count = category_read[
                (category_read['category'] == category) & 
                (category_read['is_read'] == True)
            ]['count'].sum()
            unread_count = category_read[
                (category_read['category'] == category) & 
                (category_read['is_read'] == False)
            ]['count'].sum()
            category_read_matrix.append({
                'category': category.capitalize(),
                'read': int(read_count),
                'unread': int(unread_count),
                'total': int(read_count + unread_count)
            })
        
        # 6. Hourly Distribution (When emails arrive)
        df['hour'] = df['timestamp'].dt.hour
        hourly_distribution = df.groupby('hour').size().reset_index(name='count')
        hourly_data = []
        for hour in range(24):
            count = hourly_distribution[hourly_distribution['hour'] == hour]['count'].values
            hourly_data.append({
                'hour': f"{hour:02d}:00",
                'hourNum': hour,
                'count': int(count[0]) if len(count) > 0 else 0
            })
        
        # 7. Priority vs Category Heatmap Data
        priority_category_matrix = df.groupby(['priority', 'category']).size().reset_index(name='count')
        heatmap_data = []
        for _, row in priority_category_matrix.iterrows():
            heatmap_data.append({
                'priority': row['priority'].capitalize(),
                'category': row['category'].capitalize(),
                'count': int(row['count'])
            })
        
        # 8. Email Length Analysis
        df['body_length'] = df['body'].str.len()
        df['length_category'] = pd.cut(df['body_length'], 
                                       bins=[0, 200, 500, 1000, float('inf')],
                                       labels=['Short (<200)', 'Medium (200-500)', 'Long (500-1000)', 'Very Long (>1000)'])
        length_distribution = df.groupby('length_category', observed=True).size().reset_index(name='count')
        length_distribution['length_category'] = length_distribution['length_category'].astype(str)
        
        # 9. Advanced Statistics
        total_emails = len(df)
        avg_priority_score = float(df['priority_score'].mean())
        read_percentage = float((df['is_read'].sum() / total_emails) * 100)
        high_priority_count = len(df[df['priority'] == 'high'])
        medium_priority_count = len(df[df['priority'] == 'medium'])
        low_priority_count = len(df[df['priority'] == 'low'])
        
        # 10. Day of Week Analysis
        df['day_of_week'] = df['timestamp'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_distribution = df.groupby('day_of_week').size().reset_index(name='count')
        day_distribution['day_of_week'] = pd.Categorical(day_distribution['day_of_week'], 
                                                         categories=day_order, 
                                                         ordered=True)
        day_distribution = day_distribution.sort_values('day_of_week')
        
        # 11. Category Performance Metrics
        category_metrics = df.groupby('category').agg({
            'priority_score': ['mean', 'min', 'max'],
            'is_read': 'sum',
            'id': 'count'
        }).reset_index()
        category_metrics.columns = ['category', 'avg_priority', 'min_priority', 'max_priority', 'read_count', 'total_count']
        category_metrics['read_rate'] = ((category_metrics['read_count'] / category_metrics['total_count']) * 100).round(2)
        category_metrics['category'] = category_metrics['category'].str.capitalize()
        
        # 12. Email Volume Trends (last 7 days vs previous)
        df['days_ago'] = (df['timestamp'].max() - df['timestamp']).dt.days
        last_7_days = len(df[df['days_ago'] <= 7])
        previous_7_days = len(df[(df['days_ago'] > 7) & (df['days_ago'] <= 14)])
        trend = ((last_7_days - previous_7_days) / previous_7_days * 100) if previous_7_days > 0 else 0
        
        return {
            "total_emails": total_emails,
            "summary_stats": {
                "avg_priority_score": round(avg_priority_score, 3),
                "read_percentage": round(read_percentage, 2),
                "unread_percentage": round(100 - read_percentage, 2),
                "high_priority_count": high_priority_count,
                "medium_priority_count": medium_priority_count,
                "low_priority_count": low_priority_count,
                "categories_count": len(df['category'].unique()),
                "unique_senders": len(df['sender'].unique()),
                "date_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                },
                "volume_trend": {
                    "last_7_days": last_7_days,
                    "previous_7_days": previous_7_days,
                    "change_percentage": round(trend, 2)
                }
            },
            "category_priority_breakdown": category_priority_data,
            "time_series": time_series_data,
            "priority_score_distribution": priority_distribution.to_dict('records'),
            "top_senders": sender_stats.to_dict('records'),
            "category_read_matrix": category_read_matrix,
            "hourly_distribution": hourly_data,
            "priority_category_heatmap": heatmap_data,
            "email_length_distribution": length_distribution.to_dict('records'),
            "day_of_week_distribution": day_distribution.to_dict('records'),
            "category_performance": category_metrics.to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
