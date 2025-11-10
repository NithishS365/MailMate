import { Email, DashboardStats, NotificationItem } from '../types';

// Static email data
export const staticEmails: Email[] = [
  {
    id: "email_001",
    from_address: "john.doe@company.com",
    to_address: "user@mailmate.com",
    cc_address: "team@company.com",
    subject: "Q4 Project Review Meeting",
    body: "Hi Team,\n\nI hope this email finds you well. I wanted to schedule our Q4 project review meeting to discuss the progress we've made this quarter and plan for the upcoming initiatives.\n\nProposed agenda:\n1. Review of completed milestones\n2. Budget analysis and resource allocation\n3. Timeline for Q1 2024 projects\n4. Team performance metrics\n5. Client feedback and satisfaction scores\n\nPlease let me know your availability for next week. I'm thinking Tuesday or Wednesday afternoon would work best.\n\nBest regards,\nJohn",
    timestamp: "2024-01-15T14:30:00Z",
    category: "Work",
    priority: "High",
    attachments: ["Q4_Report.pdf", "Budget_Analysis.xlsx"],
    is_read: false,
    folder: "INBOX"
  },
  {
    id: "email_002",
    from_address: "sarah.wilson@bank.com",
    to_address: "user@mailmate.com",
    subject: "Monthly Account Statement - December 2023",
    body: "Dear Valued Customer,\n\nYour monthly account statement for December 2023 is now available. Please review the attached statement for details of your account activity.\n\nAccount Summary:\n- Opening Balance: $5,247.83\n- Total Deposits: $3,200.00\n- Total Withdrawals: $1,847.92\n- Closing Balance: $6,599.91\n\nIf you have any questions about your statement, please don't hesitate to contact our customer service team.\n\nThank you for banking with us.\n\nBest regards,\nSarah Wilson\nCustomer Relations Manager",
    timestamp: "2024-01-14T09:15:00Z",
    category: "Finance",
    priority: "Medium",
    attachments: ["Statement_Dec2023.pdf"],
    is_read: true,
    folder: "INBOX"
  },
  {
    id: "email_003",
    from_address: "mom@family.com",
    to_address: "user@mailmate.com",
    subject: "Family Dinner This Sunday",
    body: "Hi Sweetheart,\n\nI hope you're doing well and not working too hard! I wanted to remind you about our family dinner this Sunday at 6 PM.\n\nYour dad is making his famous lasagna, and your sister is bringing dessert. We haven't seen you in a while, and we're all excited to catch up.\n\nPlease let me know if you can make it so I can set the table accordingly. Also, feel free to bring that nice friend of yours you mentioned!\n\nLooking forward to seeing you.\n\nLove,\nMom\n\nP.S. Don't forget to bring your appetite! 😊",
    timestamp: "2024-01-13T16:45:00Z",
    category: "Personal",
    priority: "Medium",
    attachments: [],
    is_read: false,
    folder: "INBOX"
  },
  {
    id: "email_004",
    from_address: "deals@techstore.com",
    to_address: "user@mailmate.com",
    subject: "🔥 Flash Sale: 50% Off All Electronics - Limited Time!",
    body: "FLASH SALE ALERT! 🚨\n\nDon't miss out on our biggest electronics sale of the year!\n\n⚡ 50% OFF ALL ELECTRONICS ⚡\n- Laptops starting from $299\n- Smartphones from $199\n- Tablets from $149\n- Gaming accessories from $29\n\nFREE SHIPPING on orders over $100!\n\nSale ends in 24 hours - Shop now before it's too late!\n\n[SHOP NOW] [View All Deals] [Unsubscribe]\n\nHurry, limited quantities available!\n\nTechStore Team",
    timestamp: "2024-01-13T11:20:00Z",
    category: "Shopping",
    priority: "Low",
    attachments: [],
    is_read: true,
    folder: "INBOX"
  },
  {
    id: "email_005",
    from_address: "booking@airline.com",
    to_address: "user@mailmate.com",
    subject: "Flight Confirmation - NYC to LAX",
    body: "Dear Traveler,\n\nThank you for choosing SkyHigh Airlines. Your flight has been confirmed!\n\nFlight Details:\n- Flight Number: SH1247\n- Date: January 25, 2024\n- Departure: New York (JFK) at 8:30 AM\n- Arrival: Los Angeles (LAX) at 11:45 AM PST\n- Seat: 12A (Window)\n- Confirmation Code: ABC123XYZ\n\nImportant Reminders:\n- Check-in opens 24 hours before departure\n- Arrive at airport 2 hours early for domestic flights\n- Baggage allowance: 1 carry-on + 1 personal item\n\nHave a great trip!\n\nSkyHigh Airlines Customer Service",
    timestamp: "2024-01-12T13:10:00Z",
    category: "Travel",
    priority: "High",
    attachments: ["Boarding_Pass.pdf", "Travel_Insurance.pdf"],
    is_read: false,
    folder: "INBOX"
  },
  {
    id: "email_006",
    from_address: "security@mailmate.com",
    to_address: "user@mailmate.com",
    subject: "URGENT: Suspicious Login Attempt Detected",
    body: "SECURITY ALERT\n\nWe detected a suspicious login attempt to your MailMate account:\n\nDetails:\n- Time: January 12, 2024 at 2:47 AM EST\n- Location: Unknown (IP: 192.168.1.100)\n- Device: Unknown Browser\n- Status: BLOCKED\n\nIf this was you, please ignore this message. If not, please:\n1. Change your password immediately\n2. Enable two-factor authentication\n3. Review your recent account activity\n\nYour account security is our top priority.\n\nMailMate Security Team\n[Change Password] [Account Settings]",
    timestamp: "2024-01-12T07:47:00Z",
    category: "Urgent",
    priority: "High",
    attachments: [],
    is_read: false,
    folder: "INBOX"
  },
  {
    id: "email_007",
    from_address: "newsletter@techblog.com",
    to_address: "user@mailmate.com",
    subject: "Weekly Tech Digest: AI Breakthroughs & Gadget Reviews",
    body: "🚀 WEEKLY TECH DIGEST\n\nThis Week's Highlights:\n\n🤖 AI & Machine Learning\n- OpenAI releases new GPT model with improved reasoning\n- Google's Gemini shows impressive multimodal capabilities\n- Microsoft integrates AI into Office suite\n\n📱 Gadget Reviews\n- iPhone 15 Pro Max: Camera improvements worth the upgrade?\n- Samsung Galaxy S24: AI features that actually matter\n- Apple Vision Pro: The future of mixed reality?\n\n💻 Developer News\n- React 19 beta released with new features\n- Python 3.12 performance improvements\n- GitHub Copilot gets major updates\n\nRead full articles on our website!\n\n[Read More] [Unsubscribe]",
    timestamp: "2024-01-11T08:00:00Z",
    category: "Personal",
    priority: "Low",
    attachments: [],
    is_read: true,
    folder: "INBOX"
  },
  {
    id: "email_008",
    from_address: "hr@company.com",
    to_address: "user@mailmate.com",
    cc_address: "manager@company.com",
    subject: "Performance Review Scheduled - Action Required",
    body: "Dear Team Member,\n\nYour annual performance review has been scheduled for next week. Please prepare the following materials:\n\n📋 Required Documents:\n1. Self-assessment form (attached)\n2. Goal achievement summary\n3. Professional development plan\n4. Peer feedback forms (if applicable)\n\n📅 Meeting Details:\n- Date: January 22, 2024\n- Time: 2:00 PM - 3:00 PM\n- Location: Conference Room B\n- Attendees: You, Your Manager, HR Representative\n\nPlease submit your self-assessment 48 hours before the meeting.\n\nIf you have any questions, feel free to reach out.\n\nBest regards,\nHR Department",
    timestamp: "2024-01-10T15:30:00Z",
    category: "Work",
    priority: "High",
    attachments: ["Self_Assessment_Form.pdf", "Review_Guidelines.pdf"],
    is_read: false,
    folder: "INBOX"
  },
  {
    id: "email_009",
    from_address: "support@streaming.com",
    to_address: "user@mailmate.com",
    subject: "Your Subscription Expires Soon",
    body: "Hi there!\n\nWe wanted to let you know that your StreamMax Premium subscription will expire in 7 days.\n\nCurrent Plan: Premium ($12.99/month)\nExpiration Date: January 20, 2024\n\nTo continue enjoying:\n✅ 4K Ultra HD streaming\n✅ Ad-free experience\n✅ Download for offline viewing\n✅ Access to exclusive content\n\nRenew now to avoid interruption!\n\n[Renew Subscription] [Change Plan] [Cancel]\n\nNeed help? Contact our support team 24/7.\n\nThanks for being a valued subscriber!\nStreamMax Team",
    timestamp: "2024-01-10T10:15:00Z",
    category: "Personal",
    priority: "Medium",
    attachments: [],
    is_read: true,
    folder: "INBOX"
  },
  {
    id: "email_010",
    from_address: "orders@ecommerce.com",
    to_address: "user@mailmate.com",
    subject: "Order Shipped - Tracking Information Inside",
    body: "Great news! Your order has been shipped! 📦\n\nOrder Details:\n- Order #: ORD-789456123\n- Items: Wireless Headphones, Phone Case, Screen Protector\n- Total: $89.97\n- Shipping Method: Express (2-3 business days)\n\nTracking Information:\n- Carrier: FastShip Express\n- Tracking Number: FS123456789\n- Estimated Delivery: January 15, 2024\n\n[Track Package] [View Order Details] [Contact Support]\n\nYour package is on its way! You'll receive updates as it moves through our network.\n\nThank you for shopping with us!\nE-Commerce Team",
    timestamp: "2024-01-09T14:22:00Z",
    category: "Shopping",
    priority: "Medium",
    attachments: ["Invoice_ORD789456123.pdf"],
    is_read: false,
    folder: "INBOX"
  }
];

// Static dashboard stats
export const staticDashboardStats: DashboardStats = {
  totalEmails: 1247,
  unreadEmails: 89,
  categories: ["Work", "Personal", "Finance", "Shopping", "Travel", "Urgent"],
  priorityDistribution: {
    "High": 156,
    "Medium": 743,
    "Low": 348
  },
  categoryDistribution: {
    "Work": 423,
    "Personal": 298,
    "Finance": 187,
    "Shopping": 156,
    "Travel": 98,
    "Urgent": 85
  },
  emailsPerDay: [
    { date: "2024-01-01", count: 12 },
    { date: "2024-01-02", count: 8 },
    { date: "2024-01-03", count: 15 },
    { date: "2024-01-04", count: 22 },
    { date: "2024-01-05", count: 18 },
    { date: "2024-01-06", count: 6 },
    { date: "2024-01-07", count: 4 },
    { date: "2024-01-08", count: 19 },
    { date: "2024-01-09", count: 25 },
    { date: "2024-01-10", count: 31 },
    { date: "2024-01-11", count: 28 },
    { date: "2024-01-12", count: 24 },
    { date: "2024-01-13", count: 16 },
    { date: "2024-01-14", count: 9 },
    { date: "2024-01-15", count: 21 }
  ],
  avgProcessingTime: 2.3,
  averagePerDay: 18.7,
  averageResponseTime: "2.4 hours",
  lastUpdate: new Date().toISOString()
};

// Static notifications
export const staticNotifications: NotificationItem[] = [
  {
    id: "notif_001",
    type: "email",
    title: "New urgent email received",
    message: "Security alert from MailMate Security Team",
    timestamp: "2024-01-15T14:30:00Z",
    read: false
  },
  {
    id: "notif_002",
    type: "system",
    title: "Email classification completed",
    message: "Successfully classified 25 new emails",
    timestamp: "2024-01-15T13:45:00Z",
    read: false
  },
  {
    id: "notif_003",
    type: "info",
    title: "Weekly summary ready",
    message: "Your email analytics for this week are available",
    timestamp: "2024-01-15T09:00:00Z",
    read: true
  },
  {
    id: "notif_004",
    type: "warning",
    title: "High priority email",
    message: "Performance review meeting scheduled",
    timestamp: "2024-01-14T15:30:00Z",
    read: false
  },
  {
    id: "notif_005",
    type: "success",
    title: "Backup completed",
    message: "Email data backup completed successfully",
    timestamp: "2024-01-14T02:00:00Z",
    read: true
  }
];

// Static analytics data
export const staticAnalyticsData = {
  categoryDistribution: {
    "Work": 423,
    "Personal": 298,
    "Finance": 187,
    "Shopping": 156,
    "Travel": 98,
    "Urgent": 85
  },
  priorityDistribution: {
    "High": 156,
    "Medium": 743,
    "Low": 348
  },
  timeSeriesData: [
    { date: "2024-01-01", count: 12 },
    { date: "2024-01-02", count: 8 },
    { date: "2024-01-03", count: 15 },
    { date: "2024-01-04", count: 22 },
    { date: "2024-01-05", count: 18 },
    { date: "2024-01-06", count: 6 },
    { date: "2024-01-07", count: 4 },
    { date: "2024-01-08", count: 19 },
    { date: "2024-01-09", count: 25 },
    { date: "2024-01-10", count: 31 },
    { date: "2024-01-11", count: 28 },
    { date: "2024-01-12", count: 24 },
    { date: "2024-01-13", count: 16 },
    { date: "2024-01-14", count: 9 },
    { date: "2024-01-15", count: 21 },
    { date: "2024-01-16", count: 27 },
    { date: "2024-01-17", count: 33 },
    { date: "2024-01-18", count: 29 },
    { date: "2024-01-19", count: 35 },
    { date: "2024-01-20", count: 31 },
    { date: "2024-01-21", count: 18 },
    { date: "2024-01-22", count: 14 },
    { date: "2024-01-23", count: 26 },
    { date: "2024-01-24", count: 38 },
    { date: "2024-01-25", count: 42 },
    { date: "2024-01-26", count: 36 },
    { date: "2024-01-27", count: 28 },
    { date: "2024-01-28", count: 22 },
    { date: "2024-01-29", count: 19 },
    { date: "2024-01-30", count: 25 }
  ],
  topSenders: [
    { sender: "john.doe@company.com", count: 45 },
    { sender: "hr@company.com", count: 38 },
    { sender: "newsletter@techblog.com", count: 32 },
    { sender: "support@streaming.com", count: 28 },
    { sender: "deals@techstore.com", count: 24 },
    { sender: "mom@family.com", count: 22 },
    { sender: "security@mailmate.com", count: 18 },
    { sender: "booking@airline.com", count: 15 },
    { sender: "orders@ecommerce.com", count: 12 },
    { sender: "sarah.wilson@bank.com", count: 10 }
  ],
  hourlyDistribution: [
    { hour: 0, count: 2 },
    { hour: 1, count: 1 },
    { hour: 2, count: 3 },
    { hour: 3, count: 1 },
    { hour: 4, count: 2 },
    { hour: 5, count: 4 },
    { hour: 6, count: 8 },
    { hour: 7, count: 15 },
    { hour: 8, count: 28 },
    { hour: 9, count: 45 },
    { hour: 10, count: 52 },
    { hour: 11, count: 48 },
    { hour: 12, count: 38 },
    { hour: 13, count: 42 },
    { hour: 14, count: 55 },
    { hour: 15, count: 47 },
    { hour: 16, count: 39 },
    { hour: 17, count: 32 },
    { hour: 18, count: 25 },
    { hour: 19, count: 18 },
    { hour: 20, count: 12 },
    { hour: 21, count: 8 },
    { hour: 22, count: 5 },
    { hour: 23, count: 3 }
  ],
  weeklyDistribution: [
    { day: "Monday", count: 187 },
    { day: "Tuesday", count: 203 },
    { day: "Wednesday", count: 195 },
    { day: "Thursday", count: 178 },
    { day: "Friday", count: 165 },
    { day: "Saturday", count: 89 },
    { day: "Sunday", count: 67 }
  ],
  responseTimeAnalysis: {
    averageResponseTime: "2.4 hours",
    medianResponseTime: "1.8 hours",
    responseTimeByCategory: {
      "Urgent": "0.5 hours",
      "Work": "1.2 hours",
      "Personal": "4.8 hours",
      "Finance": "2.1 hours",
      "Shopping": "6.2 hours",
      "Travel": "3.4 hours"
    }
  },
  sentimentAnalysis: {
    positive: 456,
    neutral: 623,
    negative: 168
  }
};