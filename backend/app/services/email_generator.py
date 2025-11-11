from faker import Faker
from datetime import datetime, timedelta
import random
import re

fake = Faker()

def clean_email_address(email: str) -> str:
    """Clean email address to ensure it's valid"""
    # Remove any spaces
    email = email.replace(' ', '').replace('\t', '').replace('\n', '')
    # Remove any invalid characters before @
    parts = email.split('@')
    if len(parts) == 2:
        local_part = re.sub(r'[^a-zA-Z0-9._+-]', '', parts[0])
        domain_part = parts[1]
        return f"{local_part}@{domain_part}"
    return email

# Email templates for different categories
EMAIL_TEMPLATES = {
    "work": [
        {
            "subject_templates": [
                "Project Update: Q{} Status Report",
                "Meeting Minutes - {} Team Sync",
                "Action Required: {} Review by EOD",
                "Weekly Progress Report - {}",
                "Budget Approval Needed for {}",
                "Client Presentation - {} Project",
                "Performance Review Scheduled",
                "Training Session: {} Skills Development",
                "Urgent: System Maintenance Window",
                "Team Building Event - {}"
            ],
            "body_templates": [
                "Dear Team,\n\nI wanted to provide an update on our recent progress. We've successfully completed {} phases of the project and are on track for the {} deadline. Please review the attached documentation and provide your feedback by end of week.\n\nKey highlights:\n- Completed milestone deliverables\n- Stakeholder approval received\n- Budget within allocated range\n\nLet me know if you have any questions.\n\nBest regards,\n{}",
                "Hi {},\n\nFollowing up on our discussion earlier today. The {} initiative requires your immediate attention. We need to finalize the requirements by {} to stay on schedule.\n\nAction items:\n1. Review the proposal document\n2. Schedule a follow-up meeting\n3. Coordinate with the {} team\n\nPlease confirm your availability.\n\nThanks,\n{}",
                "Hello,\n\nThis is a reminder about the upcoming {} deadline. We need to ensure all team members have completed their assigned tasks. Current status shows {}% completion.\n\nPending items:\n- Final code review\n- Documentation updates\n- Testing phase completion\n\nYour prompt response would be appreciated.\n\nRegards,\n{}"
            ]
        }
    ],
    "personal": [
        {
            "subject_templates": [
                "Weekend Plans - Let's catch up!",
                "Happy Birthday! 🎉",
                "Dinner invitation for {}",
                "Book Club Meeting Next Week",
                "Family Reunion Update",
                "Movie Night this Friday?",
                "Vacation Photos from {}",
                "Thanks for your help!",
                "Coffee catch-up soon?",
                "Holiday Plans Discussion"
            ],
            "body_templates": [
                "Hey {},\n\nHope you're doing well! I was thinking we could catch up soon. It's been too long since we last met. How about {} this weekend?\n\nLet me know what works for you!\n\nCheers,\n{}",
                "Hi there,\n\nJust wanted to share some exciting news about {}! I thought you'd be interested to hear about this. We should definitely celebrate soon.\n\nLooking forward to hearing from you.\n\nBest,\n{}",
                "Dear {},\n\nThank you so much for your help with {}. I really appreciate your support and guidance. It made a huge difference!\n\nLet's grab coffee sometime next week if you're free.\n\nWarm regards,\n{}"
            ]
        }
    ],
    "urgent": [
        {
            "subject_templates": [
                "URGENT: Action Required Immediately",
                "Critical: System Alert - {}",
                "Emergency Meeting in 30 Minutes",
                "Immediate Response Needed - {}",
                "URGENT: Security Breach Detected",
                "Critical Bug in Production",
                "Escalation: Client Issue",
                "Time-Sensitive: Contract Expiring",
                "ASAP: Executive Decision Required",
                "Priority: Server Downtime"
            ],
            "body_templates": [
                "URGENT NOTICE\n\nWe have detected a critical issue that requires immediate attention. The {} system is experiencing {} and needs to be addressed within the next hour.\n\nImmediate actions required:\n1. All hands on deck\n2. Notify relevant stakeholders\n3. Prepare incident report\n\nPlease respond ASAP.\n\n{}",
                "CRITICAL ALERT\n\nA high-priority situation has emerged regarding {}. This impacts our {} operations and requires urgent intervention.\n\nStatus: Active incident\nPriority: P1\nETA for resolution: {} hours\n\nJoin the emergency bridge immediately.\n\n{}",
                "ACTION REQUIRED NOW\n\nThe client has reported a critical issue with {}. We need to respond within the next {} minutes to maintain our SLA.\n\nPlease:\n- Stop all current work\n- Focus on this issue\n- Update every 15 minutes\n\nContact: {}\n\nUrgent regards,\n{}"
            ]
        }
    ],
    "promotion": [
        {
            "subject_templates": [
                "🎉 Exclusive {} Sale - Up to {}% Off!",
                "Limited Time Offer: {} Deal Inside",
                "Your Special Discount Code - {}",
                "Flash Sale: {} Hours Only!",
                "New Arrivals: {} Collection Now Available",
                "Member Exclusive: Early Access to {}",
                "Last Chance: {} Sale Ends Tonight!",
                "Upgrade Your {} Today - Special Pricing",
                "Free Shipping on All {} Orders",
                "Seasonal Clearance - Save Big on {}"
            ],
            "body_templates": [
                "Dear Valued Customer,\n\nWe're excited to announce our exclusive {} promotion! For a limited time, enjoy up to {}% off on selected items.\n\nOffer details:\n- Valid until: {}\n- Use code: {}\n- Free shipping on orders over ${}\n\nDon't miss out on these incredible savings!\n\nShop now: www.example.com\n\nBest regards,\n{} Team",
                "Hello {},\n\nAs a valued member, you get early access to our {} sale! Discover amazing deals on:\n\n✓ Premium products\n✓ Best-selling items\n✓ New arrivals\n\nExclusive offer: Additional {}% off with code {}\n\nThis offer expires in {} hours. Shop now to secure your favorites!\n\n{} Marketing Team",
                "Hi there,\n\nBig news! Our {} collection just launched with special introductory pricing. Plus, we're offering:\n\n🎁 Free gift with purchase\n💝 Loyalty points doubled\n📦 Express shipping available\n\nVisit our website today and use code {} at checkout for an extra {}% discount.\n\nHappy shopping!\n{} Store"
            ]
        }
    ]
}

SENDER_PROFILES = {
    "work": [
        ("Project Manager", ["sarah.johnson", "michael.chen", "david.williams", "emma.martinez"]),
        ("Team Lead", ["robert.anderson", "jennifer.taylor", "james.brown", "lisa.garcia"]),
        ("HR Department", ["hr.team", "talent.acquisition", "employee.services"]),
        ("IT Support", ["it.helpdesk", "system.admin", "tech.support"]),
        ("Executive Office", ["ceo.office", "cfo.dept", "cto.team"]),
    ],
    "personal": [
        ("Friend", ["alex.smith", "jordan.lee", "casey.wilson", "morgan.davis"]),
        ("Family", ["mom", "dad", "sister.kate", "brother.john"]),
        ("Neighbor", ["john.next door", "susan.neighbor"]),
    ],
    "urgent": [
        ("System Alert", ["system.alerts", "monitoring.team", "devops"]),
        ("Security Team", ["security.team", "incident.response"]),
        ("Client Services", ["client.escalation", "customer.success"]),
    ],
    "promotion": [
        ("Marketing", ["deals", "promotions", "offers", "newsletter"]),
        ("Store", ["shop.notifications", "sales.team", "retail.updates"]),
    ]
}

def generate_email(category: str) -> dict:
    """Generate a single AI-created professional email"""
    
    # Select templates for the category
    templates = EMAIL_TEMPLATES.get(category, EMAIL_TEMPLATES["work"])[0]
    
    # Generate subject
    subject_template = random.choice(templates["subject_templates"])
    subject = subject_template.format(
        random.choice(["the new initiative", "Q4", "the project", "this quarter", "our goals"]),
        random.choice([24, 48, 72, "tomorrow", "next week"])
    )
    
    # Generate sender
    sender_category = SENDER_PROFILES.get(category, SENDER_PROFILES["work"])
    _, sender_usernames = random.choice(sender_category)
    sender_name = fake.name()
    sender_username = random.choice(sender_usernames)
    sender_email = clean_email_address(f"{sender_username}@{'company.com' if category == 'work' else 'email.com'}")
    
    # Generate body
    body_template = random.choice(templates["body_templates"])
    body = body_template.format(
        sender_name.split()[0],
        random.choice(["development", "implementation", "deployment", "planning", "execution"]),
        random.choice(["Friday", "next Monday", "end of month", "Q4"]),
        random.choice(["engineering", "marketing", "sales", "operations"]),
        random.randint(60, 95),
        sender_name
    )
    
    # Generate timestamp (within last 30 days)
    timestamp = datetime.now() - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    return {
        "subject": subject,
        "sender": sender_name,
        "sender_email": sender_email,
        "body": body,
        "timestamp": timestamp,
        "category": category,
        "is_read": False  # All new emails start as unread
    }

def generate_50_emails() -> list:
    """Generate 50 professional emails across different categories"""
    emails = []
    
    # Distribution of emails across categories
    category_distribution = {
        "work": 20,
        "personal": 10,
        "urgent": 8,
        "promotion": 12
    }
    
    for category, count in category_distribution.items():
        for _ in range(count):
            emails.append(generate_email(category))
    
    # Sort by timestamp (newest first)
    emails.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return emails
