"""
Examples demonstrating the MailMate extensible interface system.

This module provides concrete examples of how to implement and use:
- Custom ML models
- Email providers with OAuth2
- Batch processing workflows
- Plugin development
- Configuration management

These examples serve as templates for extending MailMate functionality.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..exceptions import ClassificationError, SummarizationError, ValidationError
from ..logging_config import get_logger
from .ml_models import BaseMLModel, ClassificationModel, SummarizationModel, ModelRegistry
from .email_providers import BaseEmailProvider, OAuth2Provider, EmailProviderRegistry

# Optional batch processing imports
try:
    from .batch_processing import BatchProcessor, AsyncEmailProcessor
    HAS_BATCH_PROCESSING = True
except ImportError:
    HAS_BATCH_PROCESSING = False
    class BatchProcessor: 
        def __init__(self): pass
        async def process_batch(self, *args, **kwargs): 
            raise ImportError("Batch processing requires Celery and Redis")

from .plugin_system import PluginManager, ExtensionPoint
from .configuration import ConfigurationManager, ConfigurationProfile, ConfigurationEnvironment

logger = get_logger("mailmate.interfaces.examples")


# ============================================================================
# ML Model Examples
# ============================================================================

class SimpleClassificationModel(ClassificationModel):
    """
    Example implementation of a simple classification model.
    
    Uses basic keyword matching for demonstration purposes.
    In a real implementation, this would use actual ML algorithms.
    """
    
    def __init__(self, model_id: str = "simple_classifier"):
        super().__init__(model_id)
        self.keywords = {
            "important": ["urgent", "asap", "deadline", "critical", "priority"],
            "spam": ["offer", "deal", "discount", "free", "win", "prize"],
            "work": ["project", "meeting", "task", "deadline", "team"],
            "personal": ["family", "friend", "personal", "private"]
        }
    
    async def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Classify email text using keyword matching."""
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text input is required and must be a string")
            
            text_lower = text.lower()
            scores = {}
            
            # Score each category based on keyword matches
            for category, keywords in self.keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[category] = score / len(keywords)  # Normalize by category size
            
            # Find the category with highest score
            predicted_category = max(scores, key=scores.get)
            confidence = scores[predicted_category]
            
            result = {
                "category": predicted_category,
                "confidence": confidence,
                "scores": scores,
                "model_id": self.model_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Classified text as '{predicted_category}' with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ClassificationError(f"Simple classification failed: {str(e)}") from e
    
    async def train(self, training_data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Train the model (update keyword lists)."""
        try:
            if not training_data:
                raise ValidationError("Training data cannot be empty")
            
            # In this simple example, we just update keyword lists based on training data
            category_texts = {}
            for item in training_data:
                category = item.get("category")
                text = item.get("text", "")
                
                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(text.lower())
            
            # Extract common words for each category (simplified approach)
            for category, texts in category_texts.items():
                words = []
                for text in texts:
                    words.extend(text.split())
                
                # Keep most common words as keywords (simplified)
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                self.keywords[category] = [word for word, count in common_words[:10]]
            
            return {
                "status": "success",
                "updated_categories": list(category_texts.keys()),
                "training_samples": len(training_data),
                "model_id": self.model_id
            }
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ClassificationError(f"Training failed: {str(e)}") from e


class SimpleSummarizationModel(SummarizationModel):
    """
    Example implementation of a simple summarization model.
    
    Uses extractive summarization by selecting important sentences.
    """
    
    def __init__(self, model_id: str = "simple_summarizer"):
        super().__init__(model_id)
        self.important_words = [
            "important", "urgent", "deadline", "summary", "conclusion",
            "decision", "action", "result", "update", "announcement"
        ]
    
    async def summarize(self, text: str, max_length: int = 200, **kwargs) -> Dict[str, Any]:
        """Summarize text using simple extractive approach."""
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text input is required and must be a string")
            
            if max_length <= 0:
                raise ValidationError("max_length must be positive")
            
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if not sentences:
                return {
                    "summary": "",
                    "original_length": len(text),
                    "summary_length": 0,
                    "compression_ratio": 0.0,
                    "model_id": self.model_id
                }
            
            # Score sentences based on important words
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = sum(1 for word in self.important_words 
                           if word.lower() in sentence.lower())
                sentence_scores.append((i, sentence, score))
            
            # Sort by score and select top sentences
            sentence_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Build summary within max_length
            summary_parts = []
            current_length = 0
            
            for _, sentence, score in sentence_scores:
                sentence_with_period = sentence + '.'
                if current_length + len(sentence_with_period) <= max_length:
                    summary_parts.append(sentence_with_period)
                    current_length += len(sentence_with_period)
                else:
                    break
            
            summary = ' '.join(summary_parts)
            
            result = {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text) if text else 0.0,
                "sentences_used": len(summary_parts),
                "total_sentences": len(sentences),
                "model_id": self.model_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Generated summary: {len(summary)} chars from {len(text)} chars")
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise SummarizationError(f"Simple summarization failed: {str(e)}") from e


# ============================================================================
# Email Provider Examples
# ============================================================================

class ExampleEmailProvider(OAuth2Provider):
    """
    Example email provider implementation.
    
    Demonstrates OAuth2 authentication flow and email operations.
    """
    
    def __init__(self, provider_id: str = "example_provider"):
        super().__init__(provider_id)
        self.base_url = "https://api.example-email.com"
        
        # Example OAuth2 configuration
        self.oauth_config = {
            "authorization_url": f"{self.base_url}/oauth/authorize",
            "token_url": f"{self.base_url}/oauth/token",
            "scope": "email.read email.send",
            "redirect_uri": "http://localhost:8080/oauth/callback"
        }
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using OAuth2 flow."""
        try:
            # In a real implementation, this would make actual OAuth2 requests
            client_id = credentials.get("client_id")
            client_secret = credentials.get("client_secret")
            
            if not client_id or not client_secret:
                raise ValidationError("client_id and client_secret are required")
            
            # Simulate OAuth2 token exchange
            mock_token_response = {
                "access_token": f"mock_access_token_{client_id}",
                "refresh_token": f"mock_refresh_token_{client_id}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": self.oauth_config["scope"]
            }
            
            logger.info(f"Successfully authenticated with {self.provider_id}")
            return mock_token_response
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Authentication failed: {str(e)}") from e
    
    async def fetch_emails(
        self, 
        folder: str = "INBOX", 
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fetch emails from specified folder."""
        try:
            # In a real implementation, this would make API calls to fetch emails
            
            # Simulate email data
            mock_emails = []
            for i in range(min(limit, 10)):  # Limit to 10 for example
                email = {
                    "id": f"email_{i+1}",
                    "subject": f"Example Email {i+1}",
                    "sender": f"user{i+1}@example.com",
                    "recipient": "user@example.com",
                    "body": f"This is the body of example email {i+1}. " * 5,
                    "timestamp": datetime.now().isoformat(),
                    "folder": folder,
                    "read": i % 3 == 0,  # Mark some as read
                    "attachments": []
                }
                mock_emails.append(email)
            
            logger.info(f"Fetched {len(mock_emails)} emails from {folder}")
            return mock_emails
            
        except Exception as e:
            raise ValidationError(f"Failed to fetch emails: {str(e)}") from e
    
    async def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send an email."""
        try:
            required_fields = ["to", "subject", "body"]
            for field in required_fields:
                if field not in email_data:
                    raise ValidationError(f"Missing required field: {field}")
            
            # In a real implementation, this would send the email via API
            message_id = f"sent_{datetime.now().timestamp()}"
            
            result = {
                "message_id": message_id,
                "status": "sent",
                "timestamp": datetime.now().isoformat(),
                "provider": self.provider_id
            }
            
            logger.info(f"Sent email to {email_data['to']}: {email_data['subject']}")
            return result
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to send email: {str(e)}") from e


# ============================================================================
# Batch Processing Examples
# ============================================================================

class EmailAnalysisBatchProcessor:
    """
    Example batch processor for email analysis workflows.
    
    Demonstrates how to process large numbers of emails using
    batch processing with classification and summarization.
    """
    
    def __init__(self):
        if not HAS_BATCH_PROCESSING:
            raise ImportError(
                "Batch processing requires Celery and Redis. "
                "Install with: pip install celery redis"
            )
        
        self.processor = BatchProcessor()
        self.model_registry = ModelRegistry()
        self.provider_registry = EmailProviderRegistry()
        
        # Register example models
        async def init_models():
            await self.model_registry.register_model(SimpleClassificationModel())
            await self.model_registry.register_model(SimpleSummarizationModel())
        
        # Note: In real usage, you'd need to run this in an async context
        self._models_initialized = False
    
    async def analyze_email_batch(
        self, 
        emails: List[Dict[str, Any]],
        include_classification: bool = True,
        include_summarization: bool = True
    ) -> Dict[str, Any]:
        """Analyze a batch of emails with classification and summarization."""
        try:
            tasks = []
            
            for email in emails:
                task_data = {
                    "email": email,
                    "include_classification": include_classification,
                    "include_summarization": include_summarization
                }
                tasks.append(task_data)
            
            # Process batch
            results = await self.processor.process_batch(
                tasks,
                self._process_single_email,
                batch_size=10,
                mode="parallel"
            )
            
            # Aggregate results
            successful_results = [r for r in results if r.get("status") == "success"]
            failed_results = [r for r in results if r.get("status") == "error"]
            
            analytics = self._generate_batch_analytics(successful_results)
            
            return {
                "total_emails": len(emails),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "results": successful_results,
                "errors": failed_results,
                "analytics": analytics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch email analysis failed: {e}")
            raise
    
    async def _process_single_email(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single email within a batch."""
        try:
            email = task_data["email"]
            email_body = email.get("body", "")
            
            result = {
                "email_id": email.get("id"),
                "status": "success",
                "analysis": {}
            }
            
            # Classification
            if task_data.get("include_classification"):
                classifier = self.model_registry.get_model("simple_classifier")
                if classifier:
                    classification_result = await classifier.predict(email_body)
                    result["analysis"]["classification"] = classification_result
            
            # Summarization
            if task_data.get("include_summarization"):
                summarizer = self.model_registry.get_model("simple_summarizer")
                if summarizer:
                    summary_result = await summarizer.summarize(email_body, max_length=100)
                    result["analysis"]["summarization"] = summary_result
            
            return result
            
        except Exception as e:
            return {
                "email_id": task_data.get("email", {}).get("id"),
                "status": "error",
                "error": str(e)
            }
    
    def _generate_batch_analytics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analytics from batch processing results."""
        analytics = {
            "category_distribution": {},
            "avg_confidence": 0.0,
            "avg_compression_ratio": 0.0,
            "processing_stats": {
                "emails_classified": 0,
                "emails_summarized": 0
            }
        }
        
        classification_confidences = []
        compression_ratios = []
        
        for result in results:
            analysis = result.get("analysis", {})
            
            # Classification analytics
            if "classification" in analysis:
                analytics["processing_stats"]["emails_classified"] += 1
                classification = analysis["classification"]
                
                category = classification.get("category")
                confidence = classification.get("confidence", 0)
                
                if category:
                    analytics["category_distribution"][category] = (
                        analytics["category_distribution"].get(category, 0) + 1
                    )
                
                if confidence:
                    classification_confidences.append(confidence)
            
            # Summarization analytics
            if "summarization" in analysis:
                analytics["processing_stats"]["emails_summarized"] += 1
                summarization = analysis["summarization"]
                
                compression_ratio = summarization.get("compression_ratio", 0)
                if compression_ratio:
                    compression_ratios.append(compression_ratio)
        
        # Calculate averages
        if classification_confidences:
            analytics["avg_confidence"] = sum(classification_confidences) / len(classification_confidences)
        
        if compression_ratios:
            analytics["avg_compression_ratio"] = sum(compression_ratios) / len(compression_ratios)
        
        return analytics


# ============================================================================
# Plugin Example
# ============================================================================

class EmailStatisticsPlugin:
    """
    Example plugin that provides email statistics and analytics.
    
    Demonstrates how to create a plugin that integrates with the
    MailMate system and provides additional functionality.
    """
    
    def __init__(self):
        self.plugin_id = "email_statistics"
        self.version = "1.0.0"
        self.description = "Provides detailed email statistics and analytics"
        
        # Plugin extension points
        self.extension_points = [
            "email_analyzer",
            "dashboard_widget",
            "report_generator"
        ]
    
    def initialize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the plugin with context from MailMate."""
        logger.info(f"Initializing {self.plugin_id} plugin")
        
        # Access MailMate components through context
        self.model_registry = context.get("model_registry")
        self.email_registry = context.get("email_registry")
        self.config_manager = context.get("config_manager")
        
        return {
            "status": "initialized",
            "plugin_id": self.plugin_id,
            "capabilities": ["email_analysis", "statistics", "reporting"]
        }
    
    async def analyze_email_statistics(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze email statistics."""
        if not emails:
            return {"error": "No emails provided"}
        
        stats = {
            "total_emails": len(emails),
            "senders": {},
            "subjects_by_length": {"short": 0, "medium": 0, "long": 0},
            "emails_by_hour": {},
            "attachment_stats": {"with_attachments": 0, "total_attachments": 0},
            "read_status": {"read": 0, "unread": 0}
        }
        
        for email in emails:
            # Sender statistics
            sender = email.get("sender", "unknown")
            stats["senders"][sender] = stats["senders"].get(sender, 0) + 1
            
            # Subject length analysis
            subject = email.get("subject", "")
            if len(subject) < 30:
                stats["subjects_by_length"]["short"] += 1
            elif len(subject) < 60:
                stats["subjects_by_length"]["medium"] += 1
            else:
                stats["subjects_by_length"]["long"] += 1
            
            # Time-based analysis
            timestamp = email.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    stats["emails_by_hour"][hour] = stats["emails_by_hour"].get(hour, 0) + 1
                except:
                    pass
            
            # Attachment analysis
            attachments = email.get("attachments", [])
            if attachments:
                stats["attachment_stats"]["with_attachments"] += 1
                stats["attachment_stats"]["total_attachments"] += len(attachments)
            
            # Read status
            if email.get("read", False):
                stats["read_status"]["read"] += 1
            else:
                stats["read_status"]["unread"] += 1
        
        # Calculate percentages
        total = stats["total_emails"]
        stats["percentages"] = {
            "read_percentage": (stats["read_status"]["read"] / total) * 100,
            "with_attachments_percentage": (stats["attachment_stats"]["with_attachments"] / total) * 100
        }
        
        return stats
    
    def generate_dashboard_widget(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a dashboard widget configuration."""
        return {
            "widget_type": "email_statistics",
            "title": "Email Statistics",
            "data": {
                "total_emails": stats.get("total_emails", 0),
                "top_senders": dict(list(sorted(
                    stats.get("senders", {}).items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))[:5]),
                "read_percentage": stats.get("percentages", {}).get("read_percentage", 0),
                "peak_hours": self._find_peak_hours(stats.get("emails_by_hour", {}))
            }
        }
    
    def _find_peak_hours(self, hourly_data: Dict[int, int]) -> List[int]:
        """Find the top 3 hours with most email activity."""
        if not hourly_data:
            return []
        
        sorted_hours = sorted(hourly_data.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]


# ============================================================================
# Complete Usage Example
# ============================================================================

async def comprehensive_example():
    """
    Comprehensive example demonstrating all extensible interfaces.
    
    This example shows how to:
    1. Configure the system
    2. Register ML models and email providers
    3. Process emails in batches
    4. Use plugins for analytics
    5. Manage credentials securely
    """
    logger.info("Starting comprehensive MailMate extensible interfaces example")
    
    try:
        # 1. Configuration Management
        config_manager = ConfigurationManager("example_config")
        
        # Create custom configuration profile
        custom_profile = config_manager.create_profile(
            name="example_environment",
            environment=ConfigurationEnvironment.DEVELOPMENT,
            description="Example configuration for demonstration",
            ml_models={
                "default_classifier": "simple_classifier",
                "default_summarizer": "simple_summarizer",
                "enable_caching": True
            },
            email_providers={
                "default_provider": "example_provider",
                "rate_limit": 100
            },
            batch_processing={
                "batch_size": 20,
                "max_workers": 4
            }
        )
        
        config_manager.set_active_profile("example_environment")
        
        # Store secure credentials
        config_manager.store_credential(
            "email_providers", 
            "example_provider_client_id", 
            "example_client_123",
            {"description": "OAuth2 client ID for example provider"}
        )
        
        # 2. Register ML Models
        model_registry = ModelRegistry()
        
        # Register custom models
        classifier = SimpleClassificationModel()
        summarizer = SimpleSummarizationModel()
        
        await model_registry.register_model(classifier)
        await model_registry.register_model(summarizer)
        
        # 3. Register Email Providers
        provider_registry = EmailProviderRegistry()
        
        example_provider = ExampleEmailProvider()
        await provider_registry.register_provider(example_provider)
        
        # 4. Setup Batch Processing
        batch_processor = EmailAnalysisBatchProcessor()
        
        # 5. Initialize Plugin
        plugin_manager = PluginManager()
        stats_plugin = EmailStatisticsPlugin()
        
        plugin_context = {
            "model_registry": model_registry,
            "email_registry": provider_registry,
            "config_manager": config_manager
        }
        
        await plugin_manager.register_plugin(stats_plugin, plugin_context)
        
        # 6. Simulate Email Processing Workflow
        
        # Authenticate with email provider
        credentials = {
            "client_id": config_manager.get_credential("email_providers", "example_provider_client_id"),
            "client_secret": "example_secret_456"
        }
        
        auth_result = await example_provider.authenticate(credentials)
        logger.info(f"Authentication result: {auth_result['token_type']}")
        
        # Fetch emails
        emails = await example_provider.fetch_emails(folder="INBOX", limit=25)
        logger.info(f"Fetched {len(emails)} emails")
        
        # Process emails in batch
        analysis_results = await batch_processor.analyze_email_batch(
            emails,
            include_classification=True,
            include_summarization=True
        )
        
        logger.info(f"Batch analysis: {analysis_results['successful']} successful, "
                   f"{analysis_results['failed']} failed")
        
        # Generate statistics using plugin
        email_stats = await stats_plugin.analyze_email_statistics(emails)
        dashboard_widget = stats_plugin.generate_dashboard_widget(email_stats)
        
        # 7. Display Results Summary
        summary = {
            "configuration": {
                "active_profile": config_manager.get_active_profile().name,
                "total_profiles": len(config_manager._profiles)
            },
            "models": {
                "registered_models": len(model_registry._models),
                "available_models": list(model_registry._models.keys())
            },
            "providers": {
                "registered_providers": len(provider_registry._providers),
                "available_providers": list(provider_registry._providers.keys())
            },
            "processing": {
                "emails_processed": analysis_results["total_emails"],
                "success_rate": f"{(analysis_results['successful'] / analysis_results['total_emails']) * 100:.1f}%"
            },
            "analytics": {
                "total_senders": len(email_stats["senders"]),
                "most_active_sender": max(email_stats["senders"].items(), key=lambda x: x[1])[0] if email_stats["senders"] else "None",
                "read_percentage": f"{email_stats['percentages']['read_percentage']:.1f}%"
            }
        }
        
        logger.info("Example completed successfully!")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive example
    asyncio.run(comprehensive_example())