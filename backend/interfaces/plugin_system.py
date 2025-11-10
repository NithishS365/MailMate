"""
Plugin System for MailMate

This module provides a comprehensive plugin architecture for dynamically
loading and managing ML models, email providers, and other extensions.
Features include:

- Dynamic plugin discovery and loading
- Plugin metadata and versioning
- Extension points for core functionality
- Plugin lifecycle management
- Dependency resolution and validation
- Plugin configuration and settings

Designed for easy extensibility and modular architecture.
"""

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Union, Type, Callable,
    Protocol, runtime_checkable, get_type_hints
)
import logging
import json
import yaml
from packaging import version

from ..exceptions import (
    ValidationError, ConfigurationError, MailMateError,
    validate_string, validate_dict, validate_list
)
from ..logging_config import get_logger

logger = get_logger("mailmate.interfaces.plugin_system")


class PluginType(Enum):
    """Types of plugins supported."""
    ML_MODEL = "ml_model"
    EMAIL_PROVIDER = "email_provider"
    DATA_EXPORTER = "data_exporter"
    AUTHENTICATION = "authentication"
    TASK_PROCESSOR = "task_processor"
    UI_COMPONENT = "ui_component"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Plugin status indicators."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """
    Metadata for a plugin with validation.
    
    Contains all information needed to identify, load, and manage a plugin.
    """
    plugin_id: str
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    email: str = ""
    website: str = ""
    license: str = ""
    entry_point: str = ""
    module_path: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    min_mailmate_version: str = "1.0.0"
    max_mailmate_version: str = "*"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        self._validate_metadata()
    
    def _validate_metadata(self) -> None:
        """Validate plugin metadata."""
        try:
            # Validate required string fields
            validate_string(self.plugin_id, "plugin_id", min_length=1, max_length=100)
            validate_string(self.name, "name", min_length=1, max_length=200)
            validate_string(self.version, "version", min_length=1, max_length=50)
            
            # Validate plugin type
            if not isinstance(self.plugin_type, PluginType):
                raise ValidationError("plugin_type must be a PluginType enum value")
            
            # Validate version format
            try:
                version.parse(self.version)
            except Exception as e:
                raise ValidationError(f"Invalid version format: {self.version}") from e
            
            # Validate entry point
            if self.entry_point:
                validate_string(self.entry_point, "entry_point", min_length=1, max_length=500)
            
            # Validate dependencies
            validate_list(self.dependencies, "dependencies")
            for i, dep in enumerate(self.dependencies):
                validate_string(dep, f"dependencies[{i}]", min_length=1, max_length=200)
            
            # Validate config schema
            validate_dict(self.config_schema, "config_schema")
            
            # Validate tags
            validate_list(self.tags, "tags")
            for i, tag in enumerate(self.tags):
                validate_string(tag, f"tags[{i}]", min_length=1, max_length=50)
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Plugin metadata validation failed: {str(e)}") from e
    
    def is_compatible_with_version(self, mailmate_version: str) -> bool:
        """Check if plugin is compatible with MailMate version."""
        try:
            current = version.parse(mailmate_version)
            min_ver = version.parse(self.min_mailmate_version)
            
            if self.max_mailmate_version == "*":
                return current >= min_ver
            
            max_ver = version.parse(self.max_mailmate_version)
            return min_ver <= current <= max_ver
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        data = {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type.value,
            "description": self.description,
            "author": self.author,
            "email": self.email,
            "website": self.website,
            "license": self.license,
            "entry_point": self.entry_point,
            "module_path": self.module_path,
            "config_schema": self.config_schema,
            "dependencies": self.dependencies,
            "min_mailmate_version": self.min_mailmate_version,
            "max_mailmate_version": self.max_mailmate_version,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create metadata from dictionary."""
        # Convert plugin_type from string to enum
        if 'plugin_type' in data and isinstance(data['plugin_type'], str):
            data['plugin_type'] = PluginType(data['plugin_type'])
        
        # Convert created_at from string to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@runtime_checkable
class PluginInterface(Protocol):
    """Protocol that all plugins must implement."""
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        ...
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        ...
    
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        ...


class ExtensionPoint:
    """
    Extension point for plugin integration.
    
    Defines where and how plugins can extend core functionality.
    """
    
    def __init__(
        self, 
        name: str,
        description: str,
        interface_type: Type,
        required_methods: List[str] = None
    ) -> None:
        """
        Initialize extension point.
        
        Args:
            name: Extension point name
            description: Description of what this extension point does
            interface_type: Expected interface/protocol type
            required_methods: List of required methods
        """
        validate_string(name, "name", min_length=1, max_length=100)
        validate_string(description, "description", min_length=1)
        
        self.name = name
        self.description = description
        self.interface_type = interface_type
        self.required_methods = required_methods or []
        self._plugins: Dict[str, Any] = {}
        self.logger = get_logger(f"mailmate.extension.{name}")
    
    def register_plugin(self, plugin_id: str, plugin_instance: Any) -> bool:
        """
        Register a plugin at this extension point.
        
        Args:
            plugin_id: Plugin identifier
            plugin_instance: Plugin instance
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate plugin implements required interface
            if not isinstance(plugin_instance, self.interface_type):
                raise ValidationError(f"Plugin {plugin_id} does not implement required interface {self.interface_type}")
            
            # Check required methods
            for method_name in self.required_methods:
                if not hasattr(plugin_instance, method_name):
                    raise ValidationError(f"Plugin {plugin_id} missing required method: {method_name}")
            
            self._plugins[plugin_id] = plugin_instance
            self.logger.info(f"Registered plugin {plugin_id} at extension point {self.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin_id}: {e}")
            return False
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin from this extension point."""
        if plugin_id in self._plugins:
            del self._plugins[plugin_id]
            self.logger.info(f"Unregistered plugin {plugin_id} from extension point {self.name}")
            return True
        return False
    
    def get_plugin(self, plugin_id: str) -> Optional[Any]:
        """Get a registered plugin."""
        return self._plugins.get(plugin_id)
    
    def list_plugins(self) -> List[str]:
        """List registered plugin IDs."""
        return list(self._plugins.keys())
    
    def call_plugin_method(self, plugin_id: str, method_name: str, *args, **kwargs) -> Any:
        """Call a method on a registered plugin."""
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            raise ValidationError(f"Plugin {plugin_id} not registered at extension point {self.name}")
        
        if not hasattr(plugin, method_name):
            raise ValidationError(f"Plugin {plugin_id} does not have method {method_name}")
        
        method = getattr(plugin, method_name)
        return method(*args, **kwargs)


class PluginLoader:
    """
    Plugin loading and management system.
    
    Handles discovery, loading, validation, and lifecycle management of plugins.
    """
    
    def __init__(self, plugin_directories: List[str] = None) -> None:
        """
        Initialize plugin loader.
        
        Args:
            plugin_directories: List of directories to search for plugins
        """
        self.plugin_directories = plugin_directories or ["plugins", "extensions"]
        self.logger = get_logger("mailmate.plugin_loader")
        
        # Convert to Path objects and validate
        self.plugin_paths = []
        for directory in self.plugin_directories:
            path = Path(directory)
            if path.exists() and path.is_dir():
                self.plugin_paths.append(path)
            else:
                self.logger.warning(f"Plugin directory does not exist: {directory}")
    
    def discover_plugins(self) -> Dict[str, PluginMetadata]:
        """
        Discover plugins in configured directories.
        
        Returns:
            Dict[str, PluginMetadata]: Map of plugin_id to metadata
        """
        discovered_plugins = {}
        
        for plugin_path in self.plugin_paths:
            try:
                # Look for plugin.yaml or plugin.json files
                for metadata_file in plugin_path.rglob("plugin.*"):
                    if metadata_file.suffix in [".yaml", ".yml", ".json"]:
                        try:
                            metadata = self._load_plugin_metadata(metadata_file)
                            if metadata:
                                discovered_plugins[metadata.plugin_id] = metadata
                                self.logger.info(f"Discovered plugin: {metadata.plugin_id}")
                        except Exception as e:
                            self.logger.error(f"Failed to load plugin metadata from {metadata_file}: {e}")
            
            except Exception as e:
                self.logger.error(f"Error discovering plugins in {plugin_path}: {e}")
        
        return discovered_plugins
    
    def _load_plugin_metadata(self, metadata_file: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from file."""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                if metadata_file.suffix == '.json':
                    data = json.load(f)
                else:  # YAML
                    data = yaml.safe_load(f)
            
            # Add module path relative to plugin directory
            data['module_path'] = str(metadata_file.parent)
            
            return PluginMetadata.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to parse plugin metadata {metadata_file}: {e}")
            return None
    
    def load_plugin(self, metadata: PluginMetadata) -> Optional[Any]:
        """
        Load a plugin from its metadata.
        
        Args:
            metadata: Plugin metadata
            
        Returns:
            Plugin instance or None if loading failed
        """
        try:
            # Validate dependencies first
            if not self._check_dependencies(metadata.dependencies):
                raise ConfigurationError(f"Plugin {metadata.plugin_id} has unmet dependencies")
            
            # Load the plugin module
            plugin_module = self._load_plugin_module(metadata)
            if not plugin_module:
                raise ConfigurationError(f"Failed to load plugin module for {metadata.plugin_id}")
            
            # Get the plugin class from entry point
            plugin_class = self._get_plugin_class(plugin_module, metadata.entry_point)
            if not plugin_class:
                raise ConfigurationError(f"Failed to find plugin class {metadata.entry_point} in {metadata.plugin_id}")
            
            # Instantiate the plugin
            plugin_instance = plugin_class()
            
            # Validate plugin implements required interface
            if not isinstance(plugin_instance, PluginInterface):
                self.logger.warning(f"Plugin {metadata.plugin_id} does not implement PluginInterface")
            
            self.logger.info(f"Loaded plugin: {metadata.plugin_id}")
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {metadata.plugin_id}: {e}")
            return None
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dependency in dependencies:
            try:
                # Simple check - try to import the dependency
                importlib.import_module(dependency)
            except ImportError:
                self.logger.error(f"Missing dependency: {dependency}")
                return False
        return True
    
    def _load_plugin_module(self, metadata: PluginMetadata) -> Optional[Any]:
        """Load plugin module dynamically."""
        try:
            module_path = Path(metadata.module_path)
            
            # Look for __init__.py or main.py
            init_file = module_path / "__init__.py"
            main_file = module_path / "main.py"
            
            if init_file.exists():
                spec = importlib.util.spec_from_file_location(metadata.plugin_id, init_file)
            elif main_file.exists():
                spec = importlib.util.spec_from_file_location(metadata.plugin_id, main_file)
            else:
                # Try to find Python files in the directory
                python_files = list(module_path.glob("*.py"))
                if not python_files:
                    raise FileNotFoundError(f"No Python files found in {module_path}")
                
                # Use the first Python file found
                spec = importlib.util.spec_from_file_location(metadata.plugin_id, python_files[0])
            
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {metadata.plugin_id}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[metadata.plugin_id] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            self.logger.error(f"Failed to load module for plugin {metadata.plugin_id}: {e}")
            return None
    
    def _get_plugin_class(self, module: Any, entry_point: str) -> Optional[Type]:
        """Get plugin class from module."""
        try:
            if not entry_point:
                # Try to find a class that implements PluginInterface
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, PluginInterface) and obj != PluginInterface:
                        return obj
                return None
            
            # Get class by entry point
            if hasattr(module, entry_point):
                plugin_class = getattr(module, entry_point)
                if inspect.isclass(plugin_class):
                    return plugin_class
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get plugin class {entry_point}: {e}")
            return None


class PluginManager:
    """
    Central plugin management system.
    
    Coordinates plugin discovery, loading, registration, and lifecycle management.
    """
    
    def __init__(self, plugin_directories: List[str] = None) -> None:
        """Initialize plugin manager."""
        self.loader = PluginLoader(plugin_directories)
        self.logger = get_logger("mailmate.plugin_manager")
        
        # Plugin storage
        self._discovered_plugins: Dict[str, PluginMetadata] = {}
        self._loaded_plugins: Dict[str, Any] = {}
        self._plugin_status: Dict[str, PluginStatus] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # Extension points
        self._extension_points: Dict[str, ExtensionPoint] = {}
        
        # Initialize core extension points
        self._initialize_core_extension_points()
    
    def _initialize_core_extension_points(self) -> None:
        """Initialize core extension points."""
        from .ml_models import BaseMLModel
        from .email_providers import BaseEmailProvider
        
        # ML Models extension point
        self.register_extension_point(ExtensionPoint(
            name="ml_models",
            description="Machine Learning models for email processing",
            interface_type=BaseMLModel,
            required_methods=["predict", "predict_async"]
        ))
        
        # Email Providers extension point
        self.register_extension_point(ExtensionPoint(
            name="email_providers",
            description="Email provider implementations",
            interface_type=BaseEmailProvider,
            required_methods=["authenticate", "get_emails", "send_email"]
        ))
    
    def discover_all_plugins(self) -> Dict[str, PluginMetadata]:
        """Discover all plugins in configured directories."""
        self._discovered_plugins = self.loader.discover_plugins()
        
        # Update status for discovered plugins
        for plugin_id in self._discovered_plugins:
            self._plugin_status[plugin_id] = PluginStatus.DISCOVERED
        
        self.logger.info(f"Discovered {len(self._discovered_plugins)} plugins")
        return self._discovered_plugins
    
    def load_plugin(self, plugin_id: str, config: Dict[str, Any] = None) -> bool:
        """
        Load a specific plugin.
        
        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration
            
        Returns:
            bool: True if loading successful
        """
        try:
            if plugin_id not in self._discovered_plugins:
                raise ValidationError(f"Plugin {plugin_id} not discovered")
            
            metadata = self._discovered_plugins[plugin_id]
            
            # Check if already loaded
            if plugin_id in self._loaded_plugins:
                self.logger.warning(f"Plugin {plugin_id} already loaded")
                return True
            
            # Load the plugin
            plugin_instance = self.loader.load_plugin(metadata)
            if not plugin_instance:
                self._plugin_status[plugin_id] = PluginStatus.ERROR
                return False
            
            # Initialize plugin if it implements PluginInterface
            if isinstance(plugin_instance, PluginInterface):
                plugin_config = config or self._plugin_configs.get(plugin_id, {})
                if not plugin_instance.initialize(plugin_config):
                    self._plugin_status[plugin_id] = PluginStatus.ERROR
                    return False
            
            # Store loaded plugin
            self._loaded_plugins[plugin_id] = plugin_instance
            if config:
                self._plugin_configs[plugin_id] = config
            
            self._plugin_status[plugin_id] = PluginStatus.LOADED
            
            # Try to register at appropriate extension points
            self._auto_register_plugin(plugin_id, plugin_instance, metadata)
            
            self.logger.info(f"Loaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_id}: {e}")
            self._plugin_status[plugin_id] = PluginStatus.ERROR
            return False
    
    def _auto_register_plugin(self, plugin_id: str, plugin_instance: Any, metadata: PluginMetadata) -> None:
        """Automatically register plugin at appropriate extension points."""
        # Determine extension point based on plugin type
        extension_point_map = {
            PluginType.ML_MODEL: "ml_models",
            PluginType.EMAIL_PROVIDER: "email_providers"
        }
        
        extension_point_name = extension_point_map.get(metadata.plugin_type)
        if extension_point_name and extension_point_name in self._extension_points:
            extension_point = self._extension_points[extension_point_name]
            if extension_point.register_plugin(plugin_id, plugin_instance):
                self._plugin_status[plugin_id] = PluginStatus.ACTIVE
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        try:
            if plugin_id not in self._loaded_plugins:
                return False
            
            plugin_instance = self._loaded_plugins[plugin_id]
            
            # Shutdown plugin if it implements PluginInterface
            if isinstance(plugin_instance, PluginInterface):
                plugin_instance.shutdown()
            
            # Unregister from extension points
            for extension_point in self._extension_points.values():
                extension_point.unregister_plugin(plugin_id)
            
            # Remove from loaded plugins
            del self._loaded_plugins[plugin_id]
            self._plugin_status[plugin_id] = PluginStatus.DISCOVERED
            
            self.logger.info(f"Unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False
    
    def register_extension_point(self, extension_point: ExtensionPoint) -> None:
        """Register an extension point."""
        self._extension_points[extension_point.name] = extension_point
        self.logger.info(f"Registered extension point: {extension_point.name}")
    
    def get_extension_point(self, name: str) -> Optional[ExtensionPoint]:
        """Get an extension point by name."""
        return self._extension_points.get(name)
    
    def list_plugins(self, status: Optional[PluginStatus] = None) -> List[str]:
        """List plugins, optionally filtered by status."""
        if status is None:
            return list(self._discovered_plugins.keys())
        
        return [
            plugin_id for plugin_id, plugin_status in self._plugin_status.items()
            if plugin_status == status
        ]
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive plugin information."""
        if plugin_id not in self._discovered_plugins:
            return None
        
        metadata = self._discovered_plugins[plugin_id]
        status = self._plugin_status.get(plugin_id, PluginStatus.DISCOVERED)
        
        info = metadata.to_dict()
        info["status"] = status.value
        
        # Add runtime information if loaded
        if plugin_id in self._loaded_plugins:
            plugin_instance = self._loaded_plugins[plugin_id]
            if isinstance(plugin_instance, PluginInterface):
                try:
                    runtime_info = plugin_instance.get_info()
                    info["runtime_info"] = runtime_info
                except Exception as e:
                    info["runtime_error"] = str(e)
        
        return info
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        status_counts = {}
        for status in PluginStatus:
            count = sum(1 for s in self._plugin_status.values() if s == status)
            status_counts[status.value] = count
        
        return {
            "total_discovered": len(self._discovered_plugins),
            "total_loaded": len(self._loaded_plugins),
            "extension_points": len(self._extension_points),
            "status_breakdown": status_counts,
            "plugin_directories": [str(p) for p in self.loader.plugin_paths]
        }


class PluginRegistry:
    """
    Global plugin registry for easy access to loaded plugins.
    
    Provides simplified interface for common plugin operations.
    """
    
    def __init__(self) -> None:
        """Initialize plugin registry."""
        self.manager = PluginManager()
        self.logger = get_logger("mailmate.plugin_registry")
        
        # Auto-discover plugins on initialization
        self.manager.discover_all_plugins()
    
    def get_ml_model(self, model_id: str) -> Optional[Any]:
        """Get a loaded ML model plugin."""
        extension_point = self.manager.get_extension_point("ml_models")
        return extension_point.get_plugin(model_id) if extension_point else None
    
    def get_email_provider(self, provider_id: str) -> Optional[Any]:
        """Get a loaded email provider plugin."""
        extension_point = self.manager.get_extension_point("email_providers")
        return extension_point.get_plugin(provider_id) if extension_point else None
    
    def list_available_models(self) -> List[str]:
        """List available ML model plugins."""
        extension_point = self.manager.get_extension_point("ml_models")
        return extension_point.list_plugins() if extension_point else []
    
    def list_available_providers(self) -> List[str]:
        """List available email provider plugins."""
        extension_point = self.manager.get_extension_point("email_providers")
        return extension_point.list_plugins() if extension_point else []
    
    def load_plugin_by_type(self, plugin_type: PluginType) -> List[str]:
        """Load all plugins of a specific type."""
        loaded_plugins = []
        
        for plugin_id, metadata in self.manager._discovered_plugins.items():
            if metadata.plugin_type == plugin_type:
                if self.manager.load_plugin(plugin_id):
                    loaded_plugins.append(plugin_id)
        
        return loaded_plugins
    
    def auto_load_essential_plugins(self) -> Dict[str, bool]:
        """Automatically load essential plugins."""
        results = {}
        
        # Load ML model plugins
        ml_plugins = self.load_plugin_by_type(PluginType.ML_MODEL)
        for plugin_id in ml_plugins:
            results[plugin_id] = True
        
        # Load email provider plugins
        provider_plugins = self.load_plugin_by_type(PluginType.EMAIL_PROVIDER)
        for plugin_id in provider_plugins:
            results[plugin_id] = True
        
        self.logger.info(f"Auto-loaded {len(results)} essential plugins")
        return results


# Global plugin registry instance
plugin_registry = PluginRegistry()