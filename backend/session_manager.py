"""
Session State Management for MailMate Dashboard

This module provides comprehensive session state management for the dashboard:
- Browser storage management (localStorage, sessionStorage)
- Server-side session persistence via API
- User preferences and dashboard settings
- Session synchronization between browser and server
- State restoration and backup mechanisms

Enhanced with comprehensive validation, type safety, and error handling.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, TypedDict, Literal
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
import pickle
import threading
from collections import defaultdict
import os
import re

from logging_config import get_logger
from exceptions import (
    SessionError, ValidationError, MailMateError,
    validate_string, validate_dict, validate_list
)

logger = get_logger("mailmate.session")

# Type definitions for improved type safety
ThemeType = Literal["light", "dark", "auto"]
ColorScheme = Literal["blue", "green", "purple", "red", "orange"]
FontSize = Literal["small", "medium", "large", "extra-large"]
SortOrder = Literal["asc", "desc"]
NotificationFrequency = Literal["immediate", "hourly", "daily", "weekly"]

class ViewPreferences(TypedDict, total=False):
    """Type definition for view preferences."""
    emails_per_page: int
    default_sort: str
    sort_order: SortOrder
    show_preview: bool
    compact_view: bool

class DashboardLayout(TypedDict, total=False):
    """Type definition for dashboard layout."""
    sidebar_collapsed: bool
    active_tab: str
    panel_sizes: Dict[str, int]

class NotificationSettings(TypedDict, total=False):
    """Type definition for notification settings."""
    browser_notifications: bool
    email_notifications: bool
    sound_enabled: bool
    notification_frequency: NotificationFrequency

class ThemeSettings(TypedDict, total=False):
    """Type definition for theme settings."""
    theme: ThemeType
    color_scheme: ColorScheme
    font_size: FontSize
    high_contrast: bool


@dataclass
class DashboardState:
    """
    Dashboard state data structure with comprehensive validation.
    
    Attributes:
        user_id: Unique identifier for the user
        session_id: Unique session identifier
        email_filters: Filters applied to email view
        view_preferences: User interface preferences
        dashboard_layout: Dashboard layout configuration
        recent_searches: List of recent search terms
        bookmarked_emails: List of bookmarked email IDs
        notification_settings: Notification preferences
        theme_settings: UI theme configuration
        last_updated: Timestamp of last update
        expires_at: Session expiration timestamp
    """
    user_id: str
    session_id: str
    email_filters: Dict[str, Any] = field(default_factory=dict)
    view_preferences: ViewPreferences = field(default_factory=dict)
    dashboard_layout: DashboardLayout = field(default_factory=dict)
    recent_searches: List[str] = field(default_factory=list)
    bookmarked_emails: List[str] = field(default_factory=list)
    notification_settings: NotificationSettings = field(default_factory=dict)
    theme_settings: ThemeSettings = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        self._validate_state()
    
    def _validate_state(self) -> None:
        """
        Validate all state fields.
        
        Raises:
            ValidationError: If any field validation fails
        """
        try:
            # Validate required string fields
            validate_string(self.user_id, "user_id", min_length=1, max_length=100)
            validate_string(self.session_id, "session_id", min_length=1, max_length=100)
            
            # Validate UUID format for session_id
            if not self._is_valid_uuid(self.session_id):
                raise ValidationError(f"session_id must be a valid UUID: {self.session_id}")
            
            # Validate dictionaries
            validate_dict(self.email_filters, "email_filters")
            validate_dict(self.view_preferences, "view_preferences") 
            validate_dict(self.dashboard_layout, "dashboard_layout")
            validate_dict(self.notification_settings, "notification_settings")
            validate_dict(self.theme_settings, "theme_settings")
            
            # Validate lists
            validate_list(self.recent_searches, "recent_searches")
            validate_list(self.bookmarked_emails, "bookmarked_emails")
            
            # Validate list contents
            for i, search in enumerate(self.recent_searches):
                validate_string(search, f"recent_searches[{i}]", min_length=1, max_length=200)
            
            for i, email_id in enumerate(self.bookmarked_emails):
                validate_string(email_id, f"bookmarked_emails[{i}]", min_length=1, max_length=100)
            
            # Validate datetime fields
            if not isinstance(self.last_updated, datetime):
                raise ValidationError("last_updated must be a datetime object")
            
            if self.expires_at is not None and not isinstance(self.expires_at, datetime):
                raise ValidationError("expires_at must be a datetime object or None")
            
            # Validate expiration logic
            if self.expires_at and self.expires_at <= self.last_updated:
                raise ValidationError("expires_at must be after last_updated")
            
            # Validate specific field contents
            self._validate_view_preferences()
            self._validate_dashboard_layout()
            self._validate_notification_settings()
            self._validate_theme_settings()
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected validation error: {str(e)}") from e
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    def _validate_view_preferences(self) -> None:
        """Validate view preferences structure."""
        if not self.view_preferences:
            return
            
        valid_keys = {"emails_per_page", "default_sort", "sort_order", "show_preview", "compact_view"}
        invalid_keys = set(self.view_preferences.keys()) - valid_keys
        if invalid_keys:
            raise ValidationError(f"Invalid view preference keys: {invalid_keys}")
        
        # Validate specific fields
        if "emails_per_page" in self.view_preferences:
            emails_per_page = self.view_preferences["emails_per_page"]
            if not isinstance(emails_per_page, int) or emails_per_page < 1 or emails_per_page > 100:
                raise ValidationError("emails_per_page must be an integer between 1 and 100")
        
        if "sort_order" in self.view_preferences:
            if self.view_preferences["sort_order"] not in ["asc", "desc"]:
                raise ValidationError("sort_order must be 'asc' or 'desc'")
        
        for bool_field in ["show_preview", "compact_view"]:
            if bool_field in self.view_preferences:
                if not isinstance(self.view_preferences[bool_field], bool):
                    raise ValidationError(f"{bool_field} must be a boolean")
    
    def _validate_dashboard_layout(self) -> None:
        """Validate dashboard layout structure."""
        if not self.dashboard_layout:
            return
            
        valid_keys = {"sidebar_collapsed", "active_tab", "panel_sizes"}
        invalid_keys = set(self.dashboard_layout.keys()) - valid_keys
        if invalid_keys:
            raise ValidationError(f"Invalid dashboard layout keys: {invalid_keys}")
        
        if "sidebar_collapsed" in self.dashboard_layout:
            if not isinstance(self.dashboard_layout["sidebar_collapsed"], bool):
                raise ValidationError("sidebar_collapsed must be a boolean")
        
        if "active_tab" in self.dashboard_layout:
            validate_string(self.dashboard_layout["active_tab"], "active_tab", min_length=1, max_length=50)
        
        if "panel_sizes" in self.dashboard_layout:
            panel_sizes = self.dashboard_layout["panel_sizes"]
            if not isinstance(panel_sizes, dict):
                raise ValidationError("panel_sizes must be a dictionary")
            
            for key, value in panel_sizes.items():
                validate_string(key, f"panel_sizes key", min_length=1, max_length=50)
                if not isinstance(value, (int, float)) or value < 0 or value > 100:
                    raise ValidationError(f"panel_sizes[{key}] must be a number between 0 and 100")
    
    def _validate_notification_settings(self) -> None:
        """Validate notification settings structure."""
        if not self.notification_settings:
            return
            
        valid_keys = {"browser_notifications", "email_notifications", "sound_enabled", "notification_frequency"}
        invalid_keys = set(self.notification_settings.keys()) - valid_keys
        if invalid_keys:
            raise ValidationError(f"Invalid notification setting keys: {invalid_keys}")
        
        bool_fields = ["browser_notifications", "email_notifications", "sound_enabled"]
        for field in bool_fields:
            if field in self.notification_settings:
                if not isinstance(self.notification_settings[field], bool):
                    raise ValidationError(f"{field} must be a boolean")
        
        if "notification_frequency" in self.notification_settings:
            valid_frequencies = ["immediate", "hourly", "daily", "weekly"]
            if self.notification_settings["notification_frequency"] not in valid_frequencies:
                raise ValidationError(f"notification_frequency must be one of: {valid_frequencies}")
    
    def _validate_theme_settings(self) -> None:
        """Validate theme settings structure."""
        if not self.theme_settings:
            return
            
        valid_keys = {"theme", "color_scheme", "font_size", "high_contrast"}
        invalid_keys = set(self.theme_settings.keys()) - valid_keys
        if invalid_keys:
            raise ValidationError(f"Invalid theme setting keys: {invalid_keys}")
        
        if "theme" in self.theme_settings:
            valid_themes = ["light", "dark", "auto"]
            if self.theme_settings["theme"] not in valid_themes:
                raise ValidationError(f"theme must be one of: {valid_themes}")
        
        if "color_scheme" in self.theme_settings:
            valid_schemes = ["blue", "green", "purple", "red", "orange"]
            if self.theme_settings["color_scheme"] not in valid_schemes:
                raise ValidationError(f"color_scheme must be one of: {valid_schemes}")
        
        if "font_size" in self.theme_settings:
            valid_sizes = ["small", "medium", "large", "extra-large"]
            if self.theme_settings["font_size"] not in valid_sizes:
                raise ValidationError(f"font_size must be one of: {valid_sizes}")
        
        if "high_contrast" in self.theme_settings:
            if not isinstance(self.theme_settings["high_contrast"], bool):
                raise ValidationError("high_contrast must be a boolean")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dict[str, Any]: Serializable dictionary representation
            
        Raises:
            SessionError: If serialization fails
        """
        try:
            data = asdict(self)
            # Handle datetime serialization
            data['last_updated'] = self.last_updated.isoformat()
            if self.expires_at:
                data['expires_at'] = self.expires_at.isoformat()
            return data
        except Exception as e:
            raise SessionError(f"Failed to serialize DashboardState: {str(e)}") from e
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardState':
        """
        Create instance from dictionary with validation.
        
        Args:
            data: Dictionary containing state data
            
        Returns:
            DashboardState: Validated instance
            
        Raises:
            ValidationError: If data validation fails
            SessionError: If deserialization fails
        """
        try:
            # Validate input data
            validate_dict(data, "data")
            
            # Prepare data for construction
            clean_data = data.copy()
            
            # Handle datetime deserialization
            if 'last_updated' in clean_data and isinstance(clean_data['last_updated'], str):
                try:
                    clean_data['last_updated'] = datetime.fromisoformat(clean_data['last_updated'])
                except ValueError as e:
                    raise ValidationError(f"Invalid last_updated datetime format: {str(e)}") from e
            
            if 'expires_at' in clean_data and clean_data['expires_at']:
                if isinstance(clean_data['expires_at'], str):
                    try:
                        clean_data['expires_at'] = datetime.fromisoformat(clean_data['expires_at'])
                    except ValueError as e:
                        raise ValidationError(f"Invalid expires_at datetime format: {str(e)}") from e
            
            # Validate required fields
            required_fields = ['user_id', 'session_id']
            for field in required_fields:
                if field not in clean_data:
                    raise ValidationError(f"Missing required field: {field}")
            
            return cls(**clean_data)
            
        except ValidationError:
            raise
        except Exception as e:
            raise SessionError(f"Failed to deserialize DashboardState: {str(e)}") from e


class SessionManager:
    """
    Comprehensive session state management for MailMate Dashboard.
    
    Handles both browser-side storage coordination and server-side persistence
    with automatic synchronization, backup mechanisms, and robust validation.
    
    Features:
        - Input validation for all parameters
        - Type-safe operations with comprehensive error handling
        - Automatic session cleanup and expiration management
        - Session backup and restoration capabilities
        - Thread-safe operations with proper locking
        - Comprehensive logging and monitoring
    """
    
    def __init__(self, storage_dir: str = "sessions", session_timeout: int = 24*60*60) -> None:
        """
        Initialize SessionManager with validation.
        
        Args:
            storage_dir: Directory for server-side session storage
            session_timeout: Session timeout in seconds (default: 24 hours)
            
        Raises:
            ValidationError: If parameters are invalid
            SessionError: If initialization fails
        """
        try:
            # Validate inputs
            validate_string(storage_dir, "storage_dir", min_length=1, max_length=500)
            
            if not isinstance(session_timeout, int) or session_timeout < 60 or session_timeout > 7*24*60*60:
                raise ValidationError("session_timeout must be an integer between 60 seconds and 7 days")
            
            self.storage_dir = Path(storage_dir)
            self.session_timeout = session_timeout
            self.logger = get_logger("mailmate.session")
            
            # Create storage directory with proper permissions
            try:
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                # Create subdirectories
                (self.storage_dir / "backups").mkdir(exist_ok=True)
                (self.storage_dir / "temp").mkdir(exist_ok=True)
            except OSError as e:
                raise SessionError(f"Failed to create storage directory {storage_dir}: {str(e)}") from e
            
            # Validate directory permissions
            if not os.access(self.storage_dir, os.R_OK | os.W_OK):
                raise SessionError(f"Insufficient permissions for storage directory: {storage_dir}")
            
            # In-memory session cache with thread safety
            self._session_cache: Dict[str, DashboardState] = {}
            self._cache_lock = threading.RLock()
            
            # Session cleanup tracking
            self._last_cleanup = datetime.now()
            self._cleanup_interval = timedelta(hours=1)
            
            # Statistics tracking
            self._stats = {
                "sessions_created": 0,
                "sessions_loaded": 0,
                "sessions_expired": 0,
                "backup_operations": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
            
            # Load existing sessions
            self._load_sessions()
            
            self.logger.info(f"SessionManager initialized: storage={storage_dir}, timeout={session_timeout}s")
            
        except (ValidationError, SessionError):
            raise
        except Exception as e:
            raise SessionError(f"Failed to initialize SessionManager: {str(e)}") from e
    
    def _validate_session_id(self, session_id: str, param_name: str = "session_id") -> None:
        """
        Validate session ID format.
        
        Args:
            session_id: Session identifier to validate
            param_name: Parameter name for error messages
            
        Raises:
            ValidationError: If session ID is invalid
        """
        validate_string(session_id, param_name, min_length=1, max_length=100)
        
        try:
            uuid.UUID(session_id)
        except ValueError as e:
            raise ValidationError(f"{param_name} must be a valid UUID: {session_id}") from e
    
    def _validate_user_id(self, user_id: str, param_name: str = "user_id") -> None:
        """
        Validate user ID format.
        
        Args:
            user_id: User identifier to validate
            param_name: Parameter name for error messages
            
        Raises:
            ValidationError: If user ID is invalid
        """
        validate_string(user_id, param_name, min_length=1, max_length=100)
        
        # Basic sanitization check
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValidationError(f"{param_name} can only contain alphanumeric characters, underscores, and hyphens")
    
    def create_session(self, user_id: str = "default") -> DashboardState:
        """
        Create a new dashboard session with validation.
        
        Args:
            user_id: User identifier (default for demo purposes)
            
        Returns:
            DashboardState: New session state object
            
        Raises:
            ValidationError: If user_id is invalid
            SessionError: If session creation fails
        """
        try:
            # Validate input
            self._validate_user_id(user_id)
            
            # Generate session data
            session_id = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
            
            # Create state with defaults
            state = DashboardState(
                user_id=user_id,
                session_id=session_id,
                expires_at=expires_at,
                # Default validated preferences
                view_preferences={
                    "emails_per_page": 20,
                    "default_sort": "timestamp",
                    "sort_order": "desc",
                    "show_preview": True,
                    "compact_view": False
                },
                dashboard_layout={
                    "sidebar_collapsed": False,
                    "active_tab": "dashboard",
                    "panel_sizes": {
                        "email_list": 60,
                        "email_preview": 40
                    }
                },
                notification_settings={
                    "browser_notifications": True,
                    "email_notifications": True,
                    "sound_enabled": False,
                    "notification_frequency": "immediate"
                },
                theme_settings={
                    "theme": "light",
                    "color_scheme": "blue",
                    "font_size": "medium",
                    "high_contrast": False
                }
            )
            
            # Cache and persist
            with self._cache_lock:
                self._session_cache[session_id] = state
                self._stats["sessions_created"] += 1
            
            self._save_session(state)
            self.logger.info(f"Created new session: {session_id} for user: {user_id}")
            
            return state
            
        except (ValidationError, SessionError):
            raise
        except Exception as e:
            raise SessionError(f"Failed to create session for user {user_id}: {str(e)}") from e
    
    def get_session(self, session_id: str) -> Optional[DashboardState]:
        """
        Retrieve session state by ID with validation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DashboardState or None if not found/expired
            
        Raises:
            ValidationError: If session_id is invalid
            SessionError: If retrieval fails
        """
        try:
            # Validate input
            self._validate_session_id(session_id)
            
            with self._cache_lock:
                # Check cache first
                if session_id in self._session_cache:
                    state = self._session_cache[session_id]
                    self._stats["cache_hits"] += 1
                    
                    # Check if expired
                    if state.expires_at and datetime.now() > state.expires_at:
                        self.logger.info(f"Session expired: {session_id}")
                        del self._session_cache[session_id]
                        self._delete_session_file(session_id)
                        self._stats["sessions_expired"] += 1
                        return None
                    
                    return state
                else:
                    self._stats["cache_misses"] += 1
            
            # Try loading from disk
            state = self._load_session(session_id)
            if state:
                # Check expiration
                if state.expires_at and datetime.now() > state.expires_at:
                    self.logger.info(f"Session expired: {session_id}")
                    self._delete_session_file(session_id)
                    self._stats["sessions_expired"] += 1
                    return None
                
                # Add to cache
                with self._cache_lock:
                    self._session_cache[session_id] = state
                
                return state
            
            return None
            
        except ValidationError:
            raise
        except Exception as e:
            raise SessionError(f"Failed to retrieve session {session_id}: {str(e)}") from e
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session state with new data and validation.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
            
        Returns:
            bool: True if successful, False if session not found
            
        Raises:
            ValidationError: If inputs are invalid
            SessionError: If update fails
        """
        try:
            # Validate inputs
            self._validate_session_id(session_id)
            validate_dict(updates, "updates")
            
            if not updates:
                return True  # No updates to apply
            
            state = self.get_session(session_id)
            if not state:
                return False
            
            # Validate update keys
            valid_update_keys = {
                'email_filters', 'view_preferences', 'dashboard_layout',
                'recent_searches', 'bookmarked_emails', 'notification_settings',
                'theme_settings'
            }
            
            invalid_keys = set(updates.keys()) - valid_update_keys
            if invalid_keys:
                raise ValidationError(f"Invalid update keys: {invalid_keys}")
            
            # Apply updates with validation
            for key, value in updates.items():
                if not hasattr(state, key):
                    raise ValidationError(f"Invalid state field: {key}")
                
                # Validate specific field types
                if key in ['email_filters', 'view_preferences', 'dashboard_layout', 
                          'notification_settings', 'theme_settings']:
                    validate_dict(value, key)
                elif key in ['recent_searches', 'bookmarked_emails']:
                    validate_list(value, key)
                
                setattr(state, key, value)
            
            # Update timestamp
            state.last_updated = datetime.now()
            
            # Re-validate complete state
            state._validate_state()
            
            # Save changes
            with self._cache_lock:
                self._session_cache[session_id] = state
            
            self._save_session(state)
            self.logger.debug(f"Updated session: {session_id} with keys: {list(updates.keys())}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise SessionError(f"Failed to update session {session_id}: {str(e)}") from e
    
    def update_email_filters(self, session_id: str, filters: Dict[str, Any]) -> bool:
        """
        Update email filters in session state with validation.
        
        Args:
            session_id: Session identifier
            filters: Email filters to apply
            
        Returns:
            bool: True if successful
            
        Raises:
            ValidationError: If inputs are invalid
            SessionError: If update fails
        """
        try:
            self._validate_session_id(session_id)
            validate_dict(filters, "filters")
            
            return self.update_session(session_id, {"email_filters": filters})
            
        except (ValidationError, SessionError):
            raise
        except Exception as e:
            raise SessionError(f"Failed to update email filters for session {session_id}: {str(e)}") from e
    
    def update_view_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update view preferences in session state with validation.
        
        Args:
            session_id: Session identifier  
            preferences: View preferences to update
            
        Returns:
            bool: True if successful
            
        Raises:
            ValidationError: If inputs are invalid
            SessionError: If update fails
        """
        try:
            self._validate_session_id(session_id)
            validate_dict(preferences, "preferences")
            
            state = self.get_session(session_id)
            if not state:
                return False
            
            # Merge with existing preferences
            new_preferences = state.view_preferences.copy()
            new_preferences.update(preferences)
            
            # Create temporary state to validate preferences
            temp_state = DashboardState(
                user_id=state.user_id,
                session_id=state.session_id,
                view_preferences=new_preferences
            )
            temp_state._validate_view_preferences()
            
            return self.update_session(session_id, {"view_preferences": new_preferences})
            
        except (ValidationError, SessionError):
            raise
        except Exception as e:
            raise SessionError(f"Failed to update view preferences for session {session_id}: {str(e)}") from e
    
    def update_dashboard_layout(self, session_id: str, layout: Dict[str, Any]) -> bool:
        """Update dashboard layout in session state."""
        state = self.get_session(session_id)
        if not state:
            return False
        
        # Merge with existing layout
        state.dashboard_layout.update(layout)
        state.last_updated = datetime.now()
        
        with self._cache_lock:
            self._session_cache[session_id] = state
        
        self._save_session(state)
        return True
    
    def add_recent_search(self, session_id: str, search_term: str) -> bool:
        """Add search term to recent searches."""
        state = self.get_session(session_id)
        if not state:
            return False
        
        # Add to recent searches (avoid duplicates, limit to 20)
        if search_term in state.recent_searches:
            state.recent_searches.remove(search_term)
        
        state.recent_searches.insert(0, search_term)
        state.recent_searches = state.recent_searches[:20]
        state.last_updated = datetime.now()
        
        with self._cache_lock:
            self._session_cache[session_id] = state
        
        self._save_session(state)
        return True
    
    def toggle_email_bookmark(self, session_id: str, email_id: str) -> bool:
        """Toggle bookmark status for an email."""
        state = self.get_session(session_id)
        if not state:
            return False
        
        if email_id in state.bookmarked_emails:
            state.bookmarked_emails.remove(email_id)
        else:
            state.bookmarked_emails.append(email_id)
        
        state.last_updated = datetime.now()
        
        with self._cache_lock:
            self._session_cache[session_id] = state
        
        self._save_session(state)
        return True
    
    def update_theme_settings(self, session_id: str, theme_settings: Dict[str, Any]) -> bool:
        """Update theme settings in session state."""
        state = self.get_session(session_id)
        if not state:
            return False
        
        state.theme_settings.update(theme_settings)
        state.last_updated = datetime.now()
        
        with self._cache_lock:
            self._session_cache[session_id] = state
        
        self._save_session(state)
        return True
    
    def extend_session(self, session_id: str, additional_time: Optional[int] = None) -> bool:
        """
        Extend session expiration time.
        
        Args:
            session_id: Session identifier
            additional_time: Additional seconds (default: reset to full timeout)
            
        Returns:
            bool: True if successful
        """
        state = self.get_session(session_id)
        if not state:
            return False
        
        if additional_time:
            state.expires_at = datetime.now() + timedelta(seconds=additional_time)
        else:
            state.expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
        
        state.last_updated = datetime.now()
        
        with self._cache_lock:
            self._session_cache[session_id] = state
        
        self._save_session(state)
        self.logger.debug(f"Extended session: {session_id}")
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session with validation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if successful
            
        Raises:
            ValidationError: If session_id is invalid
            SessionError: If deletion fails
        """
        try:
            self._validate_session_id(session_id)
            
            with self._cache_lock:
                if session_id in self._session_cache:
                    del self._session_cache[session_id]
            
            self._delete_session_file(session_id)
            self.logger.info(f"Deleted session: {session_id}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise SessionError(f"Failed to delete session {session_id}: {str(e)}") from e
    
    def extend_session(self, session_id: str, additional_time: Optional[int] = None) -> bool:
        """
        Extend session expiration time with validation.
        
        Args:
            session_id: Session identifier
            additional_time: Additional seconds (default: reset to full timeout)
            
        Returns:
            bool: True if successful
            
        Raises:
            ValidationError: If inputs are invalid
            SessionError: If extension fails
        """
        try:
            self._validate_session_id(session_id)
            
            if additional_time is not None:
                if not isinstance(additional_time, int) or additional_time < 60:
                    raise ValidationError("additional_time must be an integer >= 60 seconds")
            
            state = self.get_session(session_id)
            if not state:
                return False
            
            if additional_time:
                state.expires_at = datetime.now() + timedelta(seconds=additional_time)
            else:
                state.expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
            
            state.last_updated = datetime.now()
            
            with self._cache_lock:
                self._session_cache[session_id] = state
            
            self._save_session(state)
            self.logger.debug(f"Extended session: {session_id}")
            
            return True
            
        except (ValidationError, SessionError):
            raise
        except Exception as e:
            raise SessionError(f"Failed to extend session {session_id}: {str(e)}") from e
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive session management statistics.
        
        Returns:
            Dict[str, Any]: Statistics and metrics
            
        Raises:
            SessionError: If statistics retrieval fails
        """
        try:
            self._cleanup_expired_sessions()
            
            with self._cache_lock:
                total_sessions = len(self._session_cache)
                user_counts = defaultdict(int)
                
                for state in self._session_cache.values():
                    user_counts[state.user_id] += 1
            
            # Calculate cache efficiency
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
            cache_hit_rate = (self._stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_active_sessions": total_sessions,
                "unique_users": len(user_counts),
                "sessions_per_user": dict(user_counts),
                "last_cleanup": self._last_cleanup.isoformat(),
                "storage_directory": str(self.storage_dir),
                "session_timeout": self.session_timeout,
                "statistics": {
                    **self._stats,
                    "cache_hit_rate_percent": round(cache_hit_rate, 2)
                }
            }
            
        except Exception as e:
            raise SessionError(f"Failed to get session statistics: {str(e)}") from e
    
    def list_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List active sessions, optionally filtered by user.
        
        Args:
            user_id: Optional user filter
            
        Returns:
            List of session summaries
        """
        self._cleanup_expired_sessions()
        
        sessions = []
        with self._cache_lock:
            for session_id, state in self._session_cache.items():
                if user_id and state.user_id != user_id:
                    continue
                
                sessions.append({
                    "session_id": session_id,
                    "user_id": state.user_id,
                    "last_updated": state.last_updated.isoformat(),
                    "expires_at": state.expires_at.isoformat() if state.expires_at else None,
                    "active": True
                })
        
        return sessions
    
    def get_browser_storage_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get state data formatted for browser storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary suitable for localStorage/sessionStorage
        """
        state = self.get_session(session_id)
        if not state:
            return {}
        
        # Create browser-friendly state
        browser_state = {
            "sessionId": session_id,
            "userId": state.user_id,
            "emailFilters": state.email_filters,
            "viewPreferences": state.view_preferences,
            "dashboardLayout": state.dashboard_layout,
            "recentSearches": state.recent_searches,
            "bookmarkedEmails": state.bookmarked_emails,
            "notificationSettings": state.notification_settings,
            "themeSettings": state.theme_settings,
            "lastUpdated": state.last_updated.isoformat(),
            "expiresAt": state.expires_at.isoformat() if state.expires_at else None
        }
        
        return browser_state
    
    def sync_from_browser_storage(
        self,
        session_id: str,
        browser_state: Dict[str, Any]
    ) -> bool:
        """
        Synchronize session state from browser storage.
        
        Args:
            session_id: Session identifier
            browser_state: State data from browser storage
            
        Returns:
            bool: True if successful
        """
        state = self.get_session(session_id)
        if not state:
            return False
        
        # Update state from browser data
        updates = {}
        
        if "emailFilters" in browser_state:
            updates["email_filters"] = browser_state["emailFilters"]
        
        if "viewPreferences" in browser_state:
            updates["view_preferences"] = browser_state["viewPreferences"]
        
        if "dashboardLayout" in browser_state:
            updates["dashboard_layout"] = browser_state["dashboardLayout"]
        
        if "recentSearches" in browser_state:
            updates["recent_searches"] = browser_state["recentSearches"]
        
        if "bookmarkedEmails" in browser_state:
            updates["bookmarked_emails"] = browser_state["bookmarkedEmails"]
        
        if "notificationSettings" in browser_state:
            updates["notification_settings"] = browser_state["notificationSettings"]
        
        if "themeSettings" in browser_state:
            updates["theme_settings"] = browser_state["themeSettings"]
        
        return self.update_session(session_id, updates)
    
    def backup_session(self, session_id: str) -> Optional[str]:
        """
        Create a backup of session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            str: Backup identifier or None if failed
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        backup_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.storage_dir / "backups" / f"{backup_id}.json"
        backup_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(backup_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            self.logger.info(f"Created session backup: {backup_id}")
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Failed to backup session {session_id}: {e}")
            return None
    
    def restore_session(self, backup_id: str) -> Optional[str]:
        """
        Restore session from backup.
        
        Args:
            backup_id: Backup identifier
            
        Returns:
            str: New session ID or None if failed
        """
        backup_path = self.storage_dir / "backups" / f"{backup_id}.json"
        
        if not backup_path.exists():
            return None
        
        try:
            with open(backup_path, 'r') as f:
                state_data = json.load(f)
            
            # Create new session from backup
            state = DashboardState.from_dict(state_data)
            state.session_id = str(uuid.uuid4())  # New session ID
            state.last_updated = datetime.now()
            state.expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
            
            with self._cache_lock:
                self._session_cache[state.session_id] = state
            
            self._save_session(state)
            self.logger.info(f"Restored session from backup: {backup_id} -> {state.session_id}")
            
            return state.session_id
            
        except Exception as e:
            self.logger.error(f"Failed to restore session from backup {backup_id}: {e}")
            return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session management statistics."""
        self._cleanup_expired_sessions()
        
        with self._cache_lock:
            total_sessions = len(self._session_cache)
            user_counts = defaultdict(int)
            
            for state in self._session_cache.values():
                user_counts[state.user_id] += 1
        
        return {
            "total_active_sessions": total_sessions,
            "unique_users": len(user_counts),
            "sessions_per_user": dict(user_counts),
            "last_cleanup": self._last_cleanup.isoformat(),
            "storage_directory": str(self.storage_dir),
            "session_timeout": self.session_timeout
        }
    
    # Enhanced private methods with validation and error handling
    
    def _save_session(self, state: DashboardState) -> None:
        """
        Save session to disk with error handling.
        
        Args:
            state: Dashboard state to save
            
        Raises:
            SessionError: If save operation fails
        """
        if not isinstance(state, DashboardState):
            raise SessionError("state must be a DashboardState instance")
        
        session_file = self.storage_dir / f"{state.session_id}.json"
        temp_file = self.storage_dir / "temp" / f"{state.session_id}.tmp"
        
        try:
            # Write to temporary file first for atomic operation
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Atomic move to final location
            temp_file.replace(session_file)
            
        except Exception as e:
            # Clean up temp file on error
            temp_file.unlink(missing_ok=True)
            raise SessionError(f"Failed to save session {state.session_id}: {str(e)}") from e
    
    def _load_session(self, session_id: str) -> Optional[DashboardState]:
        """
        Load session from disk with validation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DashboardState or None if not found
            
        Raises:
            SessionError: If load operation fails
        """
        session_file = self.storage_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Validate loaded data
            if not isinstance(state_data, dict):
                raise SessionError(f"Invalid session data format in {session_file}")
            
            state = DashboardState.from_dict(state_data)
            self._stats["sessions_loaded"] += 1
            
            return state
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in session file {session_file}: {e}")
            # Move corrupted file to avoid repeated errors
            corrupted_file = session_file.with_suffix('.corrupted')
            session_file.rename(corrupted_file)
            return None
        except (ValidationError, SessionError) as e:
            self.logger.error(f"Session validation failed for {session_id}: {e}")
            return None
        except Exception as e:
            raise SessionError(f"Failed to load session {session_id}: {str(e)}") from e
    
    def _load_sessions(self) -> None:
        """
        Load all sessions from disk into cache with validation.
        
        Raises:
            SessionError: If critical loading errors occur
        """
        if not self.storage_dir.exists():
            return
        
        loaded_count = 0
        error_count = 0
        
        try:
            for session_file in self.storage_dir.glob("*.json"):
                try:
                    session_id = session_file.stem
                    
                    # Skip if already in cache
                    if session_id in self._session_cache:
                        continue
                    
                    state = self._load_session(session_id)
                    
                    if state:
                        # Check if expired
                        if state.expires_at and datetime.now() > state.expires_at:
                            session_file.unlink(missing_ok=True)
                            self._stats["sessions_expired"] += 1
                            continue
                        
                        with self._cache_lock:
                            self._session_cache[session_id] = state
                        loaded_count += 1
                        
                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Failed to load session file {session_file}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} sessions from disk ({error_count} errors)")
            
        except Exception as e:
            raise SessionError(f"Critical error during session loading: {str(e)}") from e
    
    def _delete_session_file(self, session_id: str) -> None:
        """
        Delete session file from disk safely.
        
        Args:
            session_id: Session identifier
        """
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            session_file.unlink(missing_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to delete session file {session_id}: {e}")
    
    def _cleanup_expired_sessions(self) -> None:
        """
        Clean up expired sessions with comprehensive error handling.
        
        Raises:
            SessionError: If critical cleanup errors occur  
        """
        now = datetime.now()
        
        # Only run cleanup periodically
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_sessions = []
        
        try:
            with self._cache_lock:
                for session_id, state in self._session_cache.items():
                    if state.expires_at and now > state.expires_at:
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    try:
                        del self._session_cache[session_id]
                        self._delete_session_file(session_id)
                        self._stats["sessions_expired"] += 1
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up session {session_id}: {e}")
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            self._last_cleanup = now
            
        except Exception as e:
            # Don't raise for cleanup errors, just log
            self.logger.error(f"Error during session cleanup: {e}")


# Enhanced factory functions with validation

def create_validated_session_manager(
    storage_dir: str = "sessions", 
    session_timeout: int = 24*60*60
) -> SessionManager:
    """
    Create a SessionManager instance with validation.
    
    Args:
        storage_dir: Directory for session storage
        session_timeout: Session timeout in seconds
        
    Returns:
        SessionManager: Configured instance
        
    Raises:
        ValidationError: If parameters are invalid
        SessionError: If creation fails
    """
    try:
        return SessionManager(storage_dir=storage_dir, session_timeout=session_timeout)
    except Exception as e:
        raise SessionError(f"Failed to create SessionManager: {str(e)}") from e


def get_default_dashboard_state(user_id: str = "default") -> Dict[str, Any]:
    """
    Get default dashboard state configuration.
    
    Args:
        user_id: User identifier
        
    Returns:
        Dict[str, Any]: Default state configuration
        
    Raises:
        ValidationError: If user_id is invalid
    """
    validate_string(user_id, "user_id", min_length=1, max_length=100)
    
    return {
        "view_preferences": {
            "emails_per_page": 20,
            "default_sort": "timestamp",
            "sort_order": "desc",
            "show_preview": True,
            "compact_view": False
        },
        "dashboard_layout": {
            "sidebar_collapsed": False,
            "active_tab": "dashboard",
            "panel_sizes": {
                "email_list": 60,
                "email_preview": 40
            }
        },
        "notification_settings": {
            "browser_notifications": True,
            "email_notifications": True,
            "sound_enabled": False,
            "notification_frequency": "immediate"
        },
        "theme_settings": {
            "theme": "light",
            "color_scheme": "blue",
            "font_size": "medium",
            "high_contrast": False
        }
    }


# Global session manager instance with validation
try:
    session_manager = SessionManager()
except Exception as e:
    # Fallback for critical initialization errors
    logger.error(f"Failed to initialize global session manager: {e}")
    session_manager = None