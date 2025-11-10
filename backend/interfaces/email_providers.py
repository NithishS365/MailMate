"""
Email Provider Interfaces for MailMate

This module provides extensible interfaces for integrating various email providers
with OAuth2 authentication and other authentication methods. Supports:

- OAuth2 authentication flow
- Multiple email providers (Gmail, Outlook, Yahoo, etc.)
- Extensible provider architecture
- Token management and refresh
- Provider capabilities discovery
- Rate limiting and error handling

All providers follow a consistent interface for easy integration and testing.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Union, Tuple, Callable,
    AsyncIterator, Protocol, runtime_checkable
)
import logging
import json
import base64
import secrets
from urllib.parse import urlencode, parse_qs
import aiohttp
import jwt
from cryptography.fernet import Fernet

from ..exceptions import (
    ValidationError, APIError, ConfigurationError, MailMateError,
    validate_string, validate_dict, validate_list
)
from ..logging_config import get_logger

logger = get_logger("mailmate.interfaces.email_providers")


class AuthenticationType(Enum):
    """Supported authentication types."""
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    API_KEY = "api_key"
    APP_PASSWORD = "app_password"
    IMAP_SMTP = "imap_smtp"


class ProviderType(Enum):
    """Supported email provider types."""
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    YAHOO = "yahoo"
    IMAP_GENERIC = "imap_generic"
    EXCHANGE = "exchange"
    OFFICE365 = "office365"
    CUSTOM = "custom"


class OAuth2GrantType(Enum):
    """OAuth2 grant types."""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "device_code"


@dataclass
class AuthenticationConfig:
    """
    Authentication configuration for email providers.
    
    Supports various authentication methods with secure credential storage.
    """
    provider_id: str
    auth_type: AuthenticationType
    credentials: Dict[str, Any] = field(default_factory=dict)
    oauth2_config: Optional[Dict[str, Any]] = None
    token_storage_path: Optional[str] = None
    encryption_key: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_encryption()
    
    def _validate_config(self) -> None:
        """Validate authentication configuration."""
        try:
            validate_string(self.provider_id, "provider_id", min_length=1, max_length=100)
            
            if not isinstance(self.auth_type, AuthenticationType):
                raise ValidationError("auth_type must be an AuthenticationType enum value")
            
            validate_dict(self.credentials, "credentials")
            
            if self.oauth2_config:
                validate_dict(self.oauth2_config, "oauth2_config")
                self._validate_oauth2_config()
            
            validate_list(self.scopes, "scopes")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Authentication config validation failed: {str(e)}") from e
    
    def _validate_oauth2_config(self) -> None:
        """Validate OAuth2 specific configuration."""
        required_fields = ['client_id', 'client_secret', 'auth_url', 'token_url']
        
        for field in required_fields:
            if field not in self.oauth2_config:
                raise ValidationError(f"Missing required OAuth2 field: {field}")
            validate_string(self.oauth2_config[field], field, min_length=1)
    
    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data."""
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()
        
        try:
            self._cipher = Fernet(self.encryption_key.encode())
        except Exception as e:
            raise ConfigurationError(f"Failed to setup encryption: {str(e)}") from e
    
    def encrypt_credential(self, value: str) -> str:
        """Encrypt a credential value."""
        try:
            return self._cipher.encrypt(value.encode()).decode()
        except Exception as e:
            raise ConfigurationError(f"Failed to encrypt credential: {str(e)}") from e
    
    def decrypt_credential(self, encrypted_value: str) -> str:
        """Decrypt a credential value."""
        try:
            return self._cipher.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            raise ConfigurationError(f"Failed to decrypt credential: {str(e)}") from e
    
    def store_credential(self, key: str, value: str, encrypt: bool = True) -> None:
        """Store a credential securely."""
        if encrypt:
            self.credentials[key] = {
                'value': self.encrypt_credential(value),
                'encrypted': True
            }
        else:
            self.credentials[key] = {
                'value': value,
                'encrypted': False
            }
    
    def get_credential(self, key: str) -> Optional[str]:
        """Retrieve a credential."""
        if key not in self.credentials:
            return None
        
        cred_data = self.credentials[key]
        if isinstance(cred_data, dict):
            value = cred_data['value']
            if cred_data.get('encrypted', False):
                return self.decrypt_credential(value)
            return value
        
        # Legacy format
        return str(cred_data)


@dataclass
class ProviderCapabilities:
    """
    Defines capabilities of an email provider.
    
    Used for feature discovery and validation.
    """
    provider_type: ProviderType
    supports_oauth2: bool = False
    supports_imap: bool = False
    supports_smtp: bool = False
    supports_api: bool = False
    supports_webhooks: bool = False
    supports_push_notifications: bool = False
    max_batch_size: int = 100
    rate_limits: Dict[str, int] = field(default_factory=dict)
    supported_scopes: List[str] = field(default_factory=list)
    api_version: str = "v1"
    
    def has_capability(self, capability: str) -> bool:
        """Check if provider has a specific capability."""
        return getattr(self, f"supports_{capability}", False)


@dataclass
class OAuth2Token:
    """OAuth2 token data with automatic refresh handling."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def expires_at(self) -> datetime:
        """Calculate token expiration time."""
        return self.created_at + timedelta(seconds=self.expires_in)
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now() >= self.expires_at
    
    @property
    def expires_soon(self, threshold_minutes: int = 5) -> bool:
        """Check if token expires soon."""
        threshold = datetime.now() + timedelta(minutes=threshold_minutes)
        return self.expires_at <= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'access_token': self.access_token,
            'token_type': self.token_type,
            'expires_in': self.expires_in,
            'refresh_token': self.refresh_token,
            'scope': self.scope,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuth2Token':
        """Create from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@runtime_checkable
class EmailProviderInterface(Protocol):
    """Protocol defining the interface all email providers must implement."""
    
    async def authenticate(self) -> bool:
        """Authenticate with the provider."""
        ...
    
    async def get_emails(self, folder: str = "INBOX", limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve emails from specified folder."""
        ...
    
    async def send_email(self, email_data: Dict[str, Any]) -> bool:
        """Send an email."""
        ...
    
    async def search_emails(self, query: str, folder: str = "INBOX") -> List[Dict[str, Any]]:
        """Search emails with query."""
        ...


class BaseEmailProvider(ABC):
    """
    Abstract base class for all email providers.
    
    Provides common functionality and enforces consistent interface
    across all provider implementations.
    """
    
    def __init__(
        self, 
        auth_config: AuthenticationConfig,
        capabilities: ProviderCapabilities
    ) -> None:
        """
        Initialize the email provider.
        
        Args:
            auth_config: Authentication configuration
            capabilities: Provider capabilities
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            if not isinstance(auth_config, AuthenticationConfig):
                raise ValidationError("auth_config must be an AuthenticationConfig instance")
            
            if not isinstance(capabilities, ProviderCapabilities):
                raise ValidationError("capabilities must be a ProviderCapabilities instance")
            
            self.auth_config = auth_config
            self.capabilities = capabilities
            self.provider_id = auth_config.provider_id
            self.provider_type = capabilities.provider_type
            self.logger = get_logger(f"mailmate.provider.{self.provider_id}")
            
            # Authentication state
            self._authenticated = False
            self._current_token: Optional[OAuth2Token] = None
            self._session: Optional[aiohttp.ClientSession] = None
            
            # Rate limiting
            self._rate_limits = capabilities.rate_limits.copy()
            self._last_requests = {}
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ConfigurationError(f"Failed to initialize provider {auth_config.provider_id}: {str(e)}") from e
    
    @abstractmethod
    async def _authenticate_internal(self) -> bool:
        """Internal authentication method. Implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _get_emails_internal(self, folder: str, limit: int) -> List[Dict[str, Any]]:
        """Internal email retrieval method. Implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _send_email_internal(self, email_data: Dict[str, Any]) -> bool:
        """Internal email sending method. Implemented by subclasses."""
        pass
    
    async def authenticate(self) -> bool:
        """
        Authenticate with the email provider.
        
        Returns:
            bool: True if authentication successful
            
        Raises:
            APIError: If authentication fails
        """
        try:
            if self._authenticated and self._current_token and not self._current_token.is_expired:
                return True
            
            success = await self._authenticate_internal()
            self._authenticated = success
            
            if success:
                self.logger.info(f"Successfully authenticated with {self.provider_type.value}")
            else:
                self.logger.error(f"Authentication failed for {self.provider_type.value}")
            
            return success
            
        except Exception as e:
            self._authenticated = False
            if isinstance(e, APIError):
                raise
            raise APIError(f"Authentication failed for {self.provider_id}: {str(e)}") from e
    
    async def get_emails(self, folder: str = "INBOX", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve emails from specified folder.
        
        Args:
            folder: Folder name (default: INBOX)
            limit: Maximum number of emails to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of email data
            
        Raises:
            ValidationError: If parameters are invalid
            APIError: If retrieval fails
        """
        try:
            # Validate inputs
            validate_string(folder, "folder", min_length=1, max_length=100)
            
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                raise ValidationError("limit must be an integer between 1 and 1000")
            
            # Ensure authenticated
            if not await self.authenticate():
                raise APIError("Authentication required")
            
            # Apply rate limiting
            await self._apply_rate_limit('get_emails')
            
            # Retrieve emails
            emails = await self._get_emails_internal(folder, limit)
            
            self.logger.debug(f"Retrieved {len(emails)} emails from {folder}")
            return emails
            
        except Exception as e:
            if isinstance(e, (ValidationError, APIError)):
                raise
            raise APIError(f"Failed to retrieve emails: {str(e)}") from e
    
    async def send_email(self, email_data: Dict[str, Any]) -> bool:
        """
        Send an email.
        
        Args:
            email_data: Email data including recipients, subject, body
            
        Returns:
            bool: True if email sent successfully
            
        Raises:
            ValidationError: If email data is invalid
            APIError: If sending fails
        """
        try:
            # Validate email data
            validate_dict(email_data, "email_data")
            self._validate_email_data(email_data)
            
            # Ensure authenticated
            if not await self.authenticate():
                raise APIError("Authentication required")
            
            # Apply rate limiting
            await self._apply_rate_limit('send_email')
            
            # Send email
            success = await self._send_email_internal(email_data)
            
            if success:
                self.logger.info(f"Email sent successfully to {email_data.get('to', 'unknown')}")
            else:
                self.logger.error("Failed to send email")
            
            return success
            
        except Exception as e:
            if isinstance(e, (ValidationError, APIError)):
                raise
            raise APIError(f"Failed to send email: {str(e)}") from e
    
    async def search_emails(self, query: str, folder: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Search emails with query.
        
        Args:
            query: Search query
            folder: Folder to search in
            
        Returns:
            List[Dict[str, Any]]: List of matching emails
        """
        try:
            validate_string(query, "query", min_length=1, max_length=500)
            validate_string(folder, "folder", min_length=1, max_length=100)
            
            if not await self.authenticate():
                raise APIError("Authentication required")
            
            await self._apply_rate_limit('search_emails')
            
            # Default implementation using get_emails and filtering
            # Subclasses can override for more efficient search
            all_emails = await self.get_emails(folder, limit=1000)
            
            matching_emails = []
            query_lower = query.lower()
            
            for email in all_emails:
                # Search in subject and body
                if (query_lower in email.get('subject', '').lower() or 
                    query_lower in email.get('body', '').lower()):
                    matching_emails.append(email)
            
            return matching_emails
            
        except Exception as e:
            if isinstance(e, (ValidationError, APIError)):
                raise
            raise APIError(f"Failed to search emails: {str(e)}") from e
    
    def _validate_email_data(self, email_data: Dict[str, Any]) -> None:
        """Validate email data structure."""
        required_fields = ['to', 'subject']
        
        for field in required_fields:
            if field not in email_data:
                raise ValidationError(f"Missing required email field: {field}")
        
        # Validate recipients
        to_field = email_data['to']
        if isinstance(to_field, str):
            validate_string(to_field, "to", min_length=1)
        elif isinstance(to_field, list):
            if not to_field:
                raise ValidationError("to field cannot be empty")
            for i, recipient in enumerate(to_field):
                validate_string(recipient, f"to[{i}]", min_length=1)
        else:
            raise ValidationError("to field must be a string or list of strings")
        
        # Validate subject
        validate_string(email_data['subject'], "subject", min_length=1, max_length=1000)
    
    async def _apply_rate_limit(self, operation: str) -> None:
        """Apply rate limiting for operations."""
        if operation not in self._rate_limits:
            return
        
        limit = self._rate_limits[operation]
        now = datetime.now()
        
        # Check if we've made requests recently
        if operation in self._last_requests:
            last_request = self._last_requests[operation]
            time_diff = (now - last_request).total_seconds()
            
            # If we're hitting the rate limit, wait
            if time_diff < (60 / limit):  # Assuming limit is per minute
                wait_time = (60 / limit) - time_diff
                await asyncio.sleep(wait_time)
        
        self._last_requests[operation] = now
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        return self._session
    
    async def close(self) -> None:
        """Close provider resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        self._authenticated = False
        self.logger.info(f"Closed provider {self.provider_id}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type.value,
            "auth_type": self.auth_config.auth_type.value,
            "authenticated": self._authenticated,
            "capabilities": {
                "supports_oauth2": self.capabilities.supports_oauth2,
                "supports_api": self.capabilities.supports_api,
                "max_batch_size": self.capabilities.max_batch_size,
                "api_version": self.capabilities.api_version
            },
            "token_expires_at": self._current_token.expires_at.isoformat() if self._current_token else None
        }


class OAuth2Provider(BaseEmailProvider):
    """
    OAuth2-enabled email provider with automatic token management.
    
    Handles OAuth2 flow, token refresh, and secure token storage.
    """
    
    def __init__(
        self, 
        auth_config: AuthenticationConfig,
        capabilities: ProviderCapabilities
    ) -> None:
        """Initialize OAuth2 provider."""
        if auth_config.auth_type != AuthenticationType.OAUTH2:
            raise ValidationError("OAuth2Provider requires OAUTH2 authentication type")
        
        if not capabilities.supports_oauth2:
            raise ValidationError("Provider capabilities must support OAuth2")
        
        super().__init__(auth_config, capabilities)
        
        # OAuth2 specific configuration
        self.oauth2_config = auth_config.oauth2_config or {}
        self._validate_oauth2_setup()
    
    def _validate_oauth2_setup(self) -> None:
        """Validate OAuth2 configuration."""
        required_fields = ['client_id', 'client_secret', 'auth_url', 'token_url']
        
        for field in required_fields:
            if field not in self.oauth2_config:
                raise ConfigurationError(f"Missing OAuth2 configuration: {field}")
    
    async def _authenticate_internal(self) -> bool:
        """Internal OAuth2 authentication."""
        try:
            # Check if we have a valid token
            if self._current_token and not self._current_token.is_expired:
                return True
            
            # Try to refresh token if available
            if self._current_token and self._current_token.refresh_token:
                if await self._refresh_token():
                    return True
            
            # Start new OAuth2 flow
            return await self._start_oauth2_flow()
            
        except Exception as e:
            self.logger.error(f"OAuth2 authentication failed: {e}")
            return False
    
    async def _refresh_token(self) -> bool:
        """Refresh OAuth2 access token."""
        if not self._current_token or not self._current_token.refresh_token:
            return False
        
        try:
            session = await self.get_session()
            
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self._current_token.refresh_token,
                'client_id': self.oauth2_config['client_id'],
                'client_secret': self.oauth2_config['client_secret']
            }
            
            async with session.post(self.oauth2_config['token_url'], data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    # Update token
                    self._current_token.access_token = token_data['access_token']
                    self._current_token.expires_in = token_data.get('expires_in', 3600)
                    self._current_token.created_at = datetime.now()
                    
                    # Update refresh token if provided
                    if 'refresh_token' in token_data:
                        self._current_token.refresh_token = token_data['refresh_token']
                    
                    self.logger.info("OAuth2 token refreshed successfully")
                    return True
                else:
                    self.logger.error(f"Token refresh failed: {response.status}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            return False
    
    async def _start_oauth2_flow(self) -> bool:
        """Start OAuth2 authorization flow."""
        # This is a simplified version - in practice, you would:
        # 1. Generate authorization URL
        # 2. Redirect user to provider's auth page
        # 3. Handle callback with authorization code
        # 4. Exchange code for token
        
        # For demonstration, we'll simulate having an authorization code
        # In real implementation, this would come from the OAuth2 callback
        
        try:
            # This would normally be obtained from OAuth2 callback
            auth_code = self.auth_config.get_credential('authorization_code')
            
            if not auth_code:
                self.logger.error("No authorization code available")
                return False
            
            return await self._exchange_code_for_token(auth_code)
            
        except Exception as e:
            self.logger.error(f"OAuth2 flow failed: {e}")
            return False
    
    async def _exchange_code_for_token(self, auth_code: str) -> bool:
        """Exchange authorization code for access token."""
        try:
            session = await self.get_session()
            
            data = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'client_id': self.oauth2_config['client_id'],
                'client_secret': self.oauth2_config['client_secret'],
                'redirect_uri': self.oauth2_config.get('redirect_uri', 'http://localhost:8080/callback')
            }
            
            async with session.post(self.oauth2_config['token_url'], data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    self._current_token = OAuth2Token(
                        access_token=token_data['access_token'],
                        token_type=token_data.get('token_type', 'Bearer'),
                        expires_in=token_data.get('expires_in', 3600),
                        refresh_token=token_data.get('refresh_token'),
                        scope=token_data.get('scope')
                    )
                    
                    self.logger.info("OAuth2 token obtained successfully")
                    return True
                else:
                    self.logger.error(f"Token exchange failed: {response.status}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Token exchange error: {e}")
            return False
    
    def generate_auth_url(self, state: Optional[str] = None) -> str:
        """
        Generate OAuth2 authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            str: Authorization URL
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.oauth2_config['client_id'],
            'response_type': 'code',
            'redirect_uri': self.oauth2_config.get('redirect_uri', 'http://localhost:8080/callback'),
            'scope': ' '.join(self.auth_config.scopes),
            'state': state,
            'access_type': 'offline',  # Request refresh token
            'prompt': 'consent'
        }
        
        return f"{self.oauth2_config['auth_url']}?{urlencode(params)}"
    
    async def handle_oauth_callback(self, callback_url: str) -> bool:
        """
        Handle OAuth2 callback and extract authorization code.
        
        Args:
            callback_url: Full callback URL with parameters
            
        Returns:
            bool: True if callback handled successfully
        """
        try:
            # Parse callback URL
            if '?' not in callback_url:
                raise ValueError("Invalid callback URL")
            
            query_string = callback_url.split('?', 1)[1]
            params = parse_qs(query_string)
            
            # Check for error
            if 'error' in params:
                error = params['error'][0]
                self.logger.error(f"OAuth2 error: {error}")
                return False
            
            # Extract authorization code
            if 'code' not in params:
                raise ValueError("No authorization code in callback")
            
            auth_code = params['code'][0]
            
            # Store and exchange for token
            self.auth_config.store_credential('authorization_code', auth_code)
            return await self._exchange_code_for_token(auth_code)
            
        except Exception as e:
            self.logger.error(f"OAuth callback handling failed: {e}")
            return False


class EmailProviderRegistry:
    """
    Registry for managing email providers with thread-safe operations.
    
    Provides centralized provider management and discovery.
    """
    
    def __init__(self) -> None:
        """Initialize the provider registry."""
        self._providers: Dict[str, BaseEmailProvider] = {}
        self._provider_configs: Dict[str, AuthenticationConfig] = {}
        self._lock = asyncio.Lock()
        self.logger = get_logger("mailmate.provider_registry")
    
    async def register_provider(self, provider: BaseEmailProvider) -> None:
        """
        Register an email provider.
        
        Args:
            provider: Provider instance to register
            
        Raises:
            ValidationError: If provider is invalid
        """
        async with self._lock:
            if not isinstance(provider, BaseEmailProvider):
                raise ValidationError("provider must be a BaseEmailProvider instance")
            
            provider_id = provider.provider_id
            
            if provider_id in self._providers:
                # Close existing provider
                await self._providers[provider_id].close()
                self.logger.warning(f"Provider {provider_id} already registered, replacing")
            
            self._providers[provider_id] = provider
            self._provider_configs[provider_id] = provider.auth_config
            
            self.logger.info(f"Registered provider: {provider_id} ({provider.provider_type.value})")
    
    async def unregister_provider(self, provider_id: str) -> bool:
        """
        Unregister an email provider.
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            bool: True if provider was unregistered
        """
        async with self._lock:
            if provider_id in self._providers:
                provider = self._providers[provider_id]
                await provider.close()
                
                del self._providers[provider_id]
                del self._provider_configs[provider_id]
                
                self.logger.info(f"Unregistered provider: {provider_id}")
                return True
            
            return False
    
    def get_provider(self, provider_id: str) -> Optional[BaseEmailProvider]:
        """Get a provider by ID."""
        return self._providers.get(provider_id)
    
    def list_providers(self, provider_type: Optional[ProviderType] = None) -> List[str]:
        """List registered providers, optionally filtered by type."""
        if provider_type is None:
            return list(self._providers.keys())
        
        return [
            provider_id for provider_id, provider in self._providers.items()
            if provider.provider_type == provider_type
        ]
    
    def get_provider_info(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get provider information."""
        provider = self.get_provider(provider_id)
        return provider.get_info() if provider else None
    
    async def authenticate_all(self) -> Dict[str, bool]:
        """Authenticate all registered providers."""
        results = {}
        
        for provider_id, provider in self._providers.items():
            try:
                results[provider_id] = await provider.authenticate()
            except Exception as e:
                self.logger.error(f"Authentication failed for {provider_id}: {e}")
                results[provider_id] = False
        
        return results
    
    async def close_all(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            try:
                await provider.close()
            except Exception as e:
                self.logger.error(f"Error closing provider: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        provider_types = {}
        authenticated_count = 0
        
        for provider in self._providers.values():
            provider_type = provider.provider_type.value
            provider_types[provider_type] = provider_types.get(provider_type, 0) + 1
            
            if provider._authenticated:
                authenticated_count += 1
        
        return {
            "total_providers": len(self._providers),
            "provider_types": provider_types,
            "authenticated_providers": authenticated_count,
            "registered_providers": list(self._providers.keys())
        }


# Global provider registry instance
provider_registry = EmailProviderRegistry()