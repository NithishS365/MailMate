"""
Configuration Management for MailMate Extensible Interfaces

This module provides centralized configuration management for ML models,
email providers, batch processing, and plugins. Features include:

- Hierarchical configuration with inheritance
- Environment-specific configurations
- Configuration validation and type checking
- Hot reloading and dynamic updates
- Secure credential management
- Configuration templating and profiles

Designed for flexible and secure configuration management.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Callable, get_type_hints
import logging
import base64

# Optional imports with fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from ..exceptions import (
    ValidationError, ConfigurationError, MailMateError,
    validate_string, validate_dict, validate_list
)
from ..logging_config import get_logger

logger = get_logger("mailmate.interfaces.configuration")


class ConfigurationEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


@dataclass
class ConfigurationProfile:
    """
    Configuration profile with validation and inheritance.
    
    Profiles allow different configurations for different environments
    or use cases while maintaining consistency.
    """
    name: str
    environment: ConfigurationEnvironment
    description: str = ""
    parent_profile: Optional[str] = None
    ml_models: Dict[str, Any] = field(default_factory=dict)
    email_providers: Dict[str, Any] = field(default_factory=dict)
    batch_processing: Dict[str, Any] = field(default_factory=dict)
    plugins: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    custom: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Validate profile after initialization."""
        self._validate_profile()
    
    def _validate_profile(self) -> None:
        """Validate configuration profile."""
        try:
            validate_string(self.name, "name", min_length=1, max_length=100)
            
            if not isinstance(self.environment, ConfigurationEnvironment):
                raise ValidationError("environment must be a ConfigurationEnvironment enum value")
            
            # Validate configuration sections
            sections = [
                self.ml_models, self.email_providers, self.batch_processing,
                self.plugins, self.security, self.logging, self.custom
            ]
            
            for i, section in enumerate(sections):
                validate_dict(section, f"section_{i}")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Configuration profile validation failed: {str(e)}") from e
    
    def merge_with_parent(self, parent_profile: 'ConfigurationProfile') -> 'ConfigurationProfile':
        """
        Merge this profile with its parent profile.
        
        Args:
            parent_profile: Parent profile to merge with
            
        Returns:
            ConfigurationProfile: Merged profile
        """
        try:
            # Deep merge dictionaries
            merged_data = {
                "name": self.name,
                "environment": self.environment,
                "description": self.description,
                "parent_profile": self.parent_profile,
                "ml_models": self._deep_merge(parent_profile.ml_models, self.ml_models),
                "email_providers": self._deep_merge(parent_profile.email_providers, self.email_providers),
                "batch_processing": self._deep_merge(parent_profile.batch_processing, self.batch_processing),
                "plugins": self._deep_merge(parent_profile.plugins, self.plugins),
                "security": self._deep_merge(parent_profile.security, self.security),
                "logging": self._deep_merge(parent_profile.logging, self.logging),
                "custom": self._deep_merge(parent_profile.custom, self.custom),
                "created_at": self.created_at,
                "updated_at": datetime.now()
            }
            
            return ConfigurationProfile(**merged_data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to merge profiles: {str(e)}") from e
    
    def _deep_merge(self, parent_dict: Dict[str, Any], child_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = parent_dict.copy()
        
        for key, value in child_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "environment": self.environment.value,
            "description": self.description,
            "parent_profile": self.parent_profile,
            "ml_models": self.ml_models,
            "email_providers": self.email_providers,
            "batch_processing": self.batch_processing,
            "plugins": self.plugins,
            "security": self.security,
            "logging": self.logging,
            "custom": self.custom,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigurationProfile':
        """Create profile from dictionary."""
        # Convert environment from string to enum
        if 'environment' in data and isinstance(data['environment'], str):
            data['environment'] = ConfigurationEnvironment(data['environment'])
        
        # Convert datetime fields
        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class SecureCredentialManager:
    """
    Secure credential management with encryption.
    
    Handles sensitive configuration data like API keys, passwords,
    and OAuth tokens with proper encryption and access control.
    """
    
    def __init__(self, encryption_key: Optional[str] = None) -> None:
        """
        Initialize credential manager.
        
        Args:
            encryption_key: Base64-encoded encryption key (generated if None)
        """
        if not HAS_CRYPTOGRAPHY:
            raise ConfigurationError(
                "cryptography library not available. Install with: pip install cryptography"
            )
        
        if encryption_key:
            try:
                self._cipher = Fernet(encryption_key.encode())
            except Exception as e:
                raise ConfigurationError(f"Invalid encryption key: {str(e)}") from e
        else:
            key = Fernet.generate_key()
            self._cipher = Fernet(key)
            self._encryption_key = key.decode()
        
        self.logger = get_logger("mailmate.credential_manager")
        self._credentials: Dict[str, Dict[str, Any]] = {}
    
    def store_credential(
        self, 
        namespace: str,
        key: str, 
        value: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Store an encrypted credential.
        
        Args:
            namespace: Credential namespace (e.g., 'email_providers', 'ml_models')
            key: Credential key
            value: Credential value to encrypt
            metadata: Optional metadata
        """
        try:
            validate_string(namespace, "namespace", min_length=1, max_length=100)
            validate_string(key, "key", min_length=1, max_length=100)
            validate_string(value, "value", min_length=1)
            
            if metadata:
                validate_dict(metadata, "metadata")
            
            # Encrypt the credential value
            encrypted_value = self._cipher.encrypt(value.encode()).decode()
            
            # Store with metadata
            if namespace not in self._credentials:
                self._credentials[namespace] = {}
            
            self._credentials[namespace][key] = {
                "encrypted_value": encrypted_value,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "accessed_at": None
            }
            
            self.logger.info(f"Stored credential: {namespace}.{key}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(f"Failed to store credential {namespace}.{key}: {str(e)}") from e
    
    def get_credential(self, namespace: str, key: str) -> Optional[str]:
        """
        Retrieve and decrypt a credential.
        
        Args:
            namespace: Credential namespace
            key: Credential key
            
        Returns:
            str or None: Decrypted credential value
        """
        try:
            if namespace not in self._credentials or key not in self._credentials[namespace]:
                return None
            
            credential_data = self._credentials[namespace][key]
            encrypted_value = credential_data["encrypted_value"]
            
            # Decrypt the value
            decrypted_value = self._cipher.decrypt(encrypted_value.encode()).decode()
            
            # Update access time
            credential_data["accessed_at"] = datetime.now().isoformat()
            
            return decrypted_value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential {namespace}.{key}: {e}")
            return None
    
    def list_credentials(self, namespace: Optional[str] = None) -> Dict[str, List[str]]:
        """List available credentials."""
        if namespace:
            return {namespace: list(self._credentials.get(namespace, {}).keys())}
        
        return {ns: list(creds.keys()) for ns, creds in self._credentials.items()}
    
    def delete_credential(self, namespace: str, key: str) -> bool:
        """Delete a credential."""
        try:
            if namespace in self._credentials and key in self._credentials[namespace]:
                del self._credentials[namespace][key]
                self.logger.info(f"Deleted credential: {namespace}.{key}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete credential {namespace}.{key}: {e}")
            return False
    
    def export_credentials(self, include_values: bool = False) -> Dict[str, Any]:
        """Export credential metadata (optionally with encrypted values)."""
        export_data = {}
        
        for namespace, credentials in self._credentials.items():
            export_data[namespace] = {}
            
            for key, cred_data in credentials.items():
                if include_values:
                    export_data[namespace][key] = cred_data
                else:
                    export_data[namespace][key] = {
                        "metadata": cred_data["metadata"],
                        "created_at": cred_data["created_at"],
                        "accessed_at": cred_data["accessed_at"]
                    }
        
        return export_data


class ConfigurationManager:
    """
    Central configuration management system.
    
    Manages configuration profiles, credential storage, validation,
    and dynamic updates for all MailMate components.
    """
    
    def __init__(self, config_dir: str = "config") -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        try:
            validate_string(config_dir, "config_dir", min_length=1, max_length=500)
            
            self.config_dir = Path(config_dir)
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger = get_logger("mailmate.configuration_manager")
            
            # Configuration storage
            self._profiles: Dict[str, ConfigurationProfile] = {}
            self._active_profile: Optional[str] = None
            self._watchers: List[Callable[[str, Dict[str, Any]], None]] = []
            
            # Credential manager
            self._credential_manager = SecureCredentialManager()
            
            # Load existing configurations
            self._load_configurations()
            
            # Set default profile if none exists
            if not self._profiles:
                self._create_default_profiles()
            
            self.logger.info(f"ConfigurationManager initialized with {len(self._profiles)} profiles")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ConfigurationError(f"Failed to initialize ConfigurationManager: {str(e)}") from e
    
    def _load_configurations(self) -> None:
        """Load configuration profiles from files."""
        try:
            # Look for configuration files
            for config_file in self.config_dir.glob("*.{json,yaml,yml}"):
                try:
                    profile = self._load_profile_from_file(config_file)
                    if profile:
                        self._profiles[profile.name] = profile
                        self.logger.info(f"Loaded configuration profile: {profile.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load configuration from {config_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
    
    def _load_profile_from_file(self, config_file: Path) -> Optional[ConfigurationProfile]:
        """Load a configuration profile from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.json':
                    data = json.load(f)
                elif config_file.suffix in ['.yaml', '.yml']:
                    if not HAS_YAML:
                        self.logger.warning(f"YAML not available, skipping {config_file}")
                        return None
                    data = yaml.safe_load(f)
                else:
                    self.logger.warning(f"Unsupported config format: {config_file}")
                    return None
            
            return ConfigurationProfile.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration file {config_file}: {e}")
            return None
    
    def _create_default_profiles(self) -> None:
        """Create default configuration profiles."""
        # Development profile
        dev_profile = ConfigurationProfile(
            name="development",
            environment=ConfigurationEnvironment.DEVELOPMENT,
            description="Development environment configuration",
            ml_models={
                "default_classification_model": "sklearn_classifier",
                "default_summarization_model": "simple_summarizer",
                "model_cache_enabled": True,
                "model_timeout": 30
            },
            email_providers={
                "default_provider": "gmail",
                "oauth2_enabled": True,
                "rate_limiting": {
                    "requests_per_minute": 60,
                    "burst_limit": 10
                }
            },
            batch_processing={
                "default_batch_size": 50,
                "max_workers": 2,
                "timeout": 300,
                "redis_url": "redis://localhost:6379/0"
            },
            plugins={
                "auto_discovery": True,
                "plugin_directories": ["plugins", "extensions"],
                "auto_load_essential": True
            },
            security={
                "encryption_enabled": True,
                "token_expiry": 3600,
                "require_https": False
            },
            logging={
                "level": "DEBUG",
                "format": "detailed",
                "enable_file_logging": True
            }
        )
        
        # Production profile
        prod_profile = ConfigurationProfile(
            name="production",
            environment=ConfigurationEnvironment.PRODUCTION,
            description="Production environment configuration",
            ml_models={
                "default_classification_model": "transformer_classifier",
                "default_summarization_model": "transformer_summarizer",
                "model_cache_enabled": True,
                "model_timeout": 60
            },
            email_providers={
                "default_provider": "outlook",
                "oauth2_enabled": True,
                "rate_limiting": {
                    "requests_per_minute": 30,
                    "burst_limit": 5
                }
            },
            batch_processing={
                "default_batch_size": 100,
                "max_workers": 8,
                "timeout": 1800,
                "redis_url": "redis://redis-server:6379/0"
            },
            plugins={
                "auto_discovery": False,
                "plugin_directories": ["plugins"],
                "auto_load_essential": True
            },
            security={
                "encryption_enabled": True,
                "token_expiry": 1800,
                "require_https": True
            },
            logging={
                "level": "INFO",
                "format": "json",
                "enable_file_logging": True
            }
        )
        
        self._profiles["development"] = dev_profile
        self._profiles["production"] = prod_profile
        self._active_profile = "development"
        
        # Save default profiles
        self.save_profile("development")
        self.save_profile("production")
    
    def create_profile(
        self,
        name: str,
        environment: ConfigurationEnvironment,
        description: str = "",
        parent_profile: Optional[str] = None,
        **config_sections
    ) -> ConfigurationProfile:
        """
        Create a new configuration profile.
        
        Args:
            name: Profile name
            environment: Configuration environment
            description: Profile description
            parent_profile: Parent profile to inherit from
            **config_sections: Configuration sections (ml_models, email_providers, etc.)
            
        Returns:
            ConfigurationProfile: Created profile
        """
        try:
            validate_string(name, "name", min_length=1, max_length=100)
            
            if name in self._profiles:
                raise ValidationError(f"Profile {name} already exists")
            
            # Create profile
            profile = ConfigurationProfile(
                name=name,
                environment=environment,
                description=description,
                parent_profile=parent_profile,
                **config_sections
            )
            
            # Apply inheritance if parent specified
            if parent_profile and parent_profile in self._profiles:
                parent = self._profiles[parent_profile]
                profile = profile.merge_with_parent(parent)
            
            self._profiles[name] = profile
            self.logger.info(f"Created configuration profile: {name}")
            
            return profile
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ConfigurationError(f"Failed to create profile {name}: {str(e)}") from e
    
    def get_profile(self, name: str) -> Optional[ConfigurationProfile]:
        """Get a configuration profile by name."""
        return self._profiles.get(name)
    
    def get_active_profile(self) -> Optional[ConfigurationProfile]:
        """Get the currently active profile."""
        if self._active_profile:
            return self._profiles.get(self._active_profile)
        return None
    
    def set_active_profile(self, name: str) -> bool:
        """Set the active configuration profile."""
        if name not in self._profiles:
            return False
        
        old_profile = self._active_profile
        self._active_profile = name
        
        self.logger.info(f"Changed active profile from {old_profile} to {name}")
        
        # Notify watchers
        self._notify_watchers("profile_changed", {
            "old_profile": old_profile,
            "new_profile": name
        })
        
        return True
    
    def save_profile(self, name: str) -> bool:
        """Save a configuration profile to file."""
        try:
            if name not in self._profiles:
                return False
            
            profile = self._profiles[name]
            
            # Use JSON if YAML not available
            if HAS_YAML:
                config_file = self.config_dir / f"{name}.yaml"
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(profile.to_dict(), f, default_flow_style=False, indent=2)
            else:
                config_file = self.config_dir / f"{name}.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(profile.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Saved configuration profile: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save profile {name}: {e}")
            return False
    
    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile."""
        try:
            if name not in self._profiles:
                return False
            
            # Don't delete active profile
            if name == self._active_profile:
                raise ValidationError("Cannot delete active profile")
            
            # Remove from memory
            del self._profiles[name]
            
            # Remove file (try both formats)
            config_files = [
                self.config_dir / f"{name}.yaml",
                self.config_dir / f"{name}.yml", 
                self.config_dir / f"{name}.json"
            ]
            
            for config_file in config_files:
                if config_file.exists():
                    config_file.unlink()
                    break
            
            self.logger.info(f"Deleted configuration profile: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete profile {name}: {e}")
            return False
    
    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from the active profile.
        
        Args:
            section: Configuration section (e.g., 'ml_models')
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        active_profile = self.get_active_profile()
        if not active_profile:
            return default
        
        section_config = getattr(active_profile, section, {})
        return section_config.get(key, default)
    
    def set_config_value(self, section: str, key: str, value: Any) -> bool:
        """
        Set a configuration value in the active profile.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
            
        Returns:
            bool: True if successful
        """
        try:
            active_profile = self.get_active_profile()
            if not active_profile:
                return False
            
            section_config = getattr(active_profile, section, {})
            section_config[key] = value
            setattr(active_profile, section, section_config)
            
            # Update timestamp
            active_profile.updated_at = datetime.now()
            
            # Notify watchers
            self._notify_watchers("config_changed", {
                "section": section,
                "key": key,
                "value": value
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set config value {section}.{key}: {e}")
            return False
    
    def get_credential(self, namespace: str, key: str) -> Optional[str]:
        """Get a credential from secure storage."""
        return self._credential_manager.get_credential(namespace, key)
    
    def store_credential(
        self, 
        namespace: str, 
        key: str, 
        value: str, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Store a credential in secure storage."""
        self._credential_manager.store_credential(namespace, key, value, metadata)
    
    def add_config_watcher(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a configuration change watcher."""
        self._watchers.append(callback)
    
    def _notify_watchers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify configuration watchers of changes."""
        for watcher in self._watchers:
            try:
                watcher(event_type, data)
            except Exception as e:
                self.logger.error(f"Configuration watcher error: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get configuration manager statistics."""
        return {
            "total_profiles": len(self._profiles),
            "active_profile": self._active_profile,
            "profiles": list(self._profiles.keys()),
            "credential_namespaces": list(self._credential_manager.list_credentials().keys()),
            "watchers": len(self._watchers),
            "config_directory": str(self.config_dir)
        }


# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions for easy access
def get_config(section: str, key: str, default: Any = None) -> Any:
    """Get configuration value from active profile."""
    return config_manager.get_config_value(section, key, default)

def get_ml_model_config(key: str, default: Any = None) -> Any:
    """Get ML model configuration value."""
    return get_config("ml_models", key, default)

def get_email_provider_config(key: str, default: Any = None) -> Any:
    """Get email provider configuration value."""
    return get_config("email_providers", key, default)

def get_batch_processing_config(key: str, default: Any = None) -> Any:
    """Get batch processing configuration value."""
    return get_config("batch_processing", key, default)

def get_plugin_config(key: str, default: Any = None) -> Any:
    """Get plugin configuration value."""
    return get_config("plugins", key, default)