"""
MailMate Extensible Interfaces Package

This package provides extensible interfaces for:
- Machine Learning models (classification, summarization, etc.)
- Email providers with OAuth2 authentication
- Asynchronous batch processing capabilities
- Plugin discovery and registration
- Configuration management with secure credential storage

Designed for easy extension and integration of new models and providers.
"""

from .ml_models import (
    BaseMLModel, ClassificationModel, SummarizationModel, 
    ModelRegistry, ModelConfiguration, ModelMetrics
)
from .email_providers import (
    BaseEmailProvider, OAuth2Provider, EmailProviderRegistry,
    AuthenticationConfig, ProviderCapabilities
)

# Optional batch processing imports (requires Celery/Redis)
try:
    from .batch_processing import (
        BatchProcessor, TaskQueue, AsyncEmailProcessor,
        BatchConfiguration, ProcessingMetrics
    )
    HAS_BATCH_PROCESSING = True
except ImportError:
    HAS_BATCH_PROCESSING = False
    # Create dummy classes for type hints
    class BatchProcessor: pass
    class TaskQueue: pass
    class AsyncEmailProcessor: pass
    class BatchConfiguration: pass
    class ProcessingMetrics: pass

from .plugin_system import (
    PluginRegistry, PluginLoader, PluginMetadata,
    ExtensionPoint, PluginManager
)
from .configuration import (
    ConfigurationManager, ConfigurationProfile, SecureCredentialManager,
    ConfigurationEnvironment, config_manager, get_config,
    get_ml_model_config, get_email_provider_config, 
    get_batch_processing_config, get_plugin_config
)

__all__ = [
    # ML Models
    'BaseMLModel', 'ClassificationModel', 'SummarizationModel',
    'ModelRegistry', 'ModelConfiguration', 'ModelMetrics',
    
    # Email Providers
    'BaseEmailProvider', 'OAuth2Provider', 'EmailProviderRegistry',
    'AuthenticationConfig', 'ProviderCapabilities',
    
    # Batch Processing
    'BatchProcessor', 'TaskQueue', 'AsyncEmailProcessor',
    'BatchConfiguration', 'ProcessingMetrics',
    
    # Plugin System
    'PluginRegistry', 'PluginLoader', 'PluginMetadata',
    'ExtensionPoint', 'PluginManager',
    
    # Configuration Management
    'ConfigurationManager', 'ConfigurationProfile', 'SecureCredentialManager',
    'ConfigurationEnvironment', 'config_manager', 'get_config',
    'get_ml_model_config', 'get_email_provider_config', 
    'get_batch_processing_config', 'get_plugin_config'
]