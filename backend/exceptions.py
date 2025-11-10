#!/usr/bin/env python3
"""
Custom Exception Classes for MailMate Application

This module defines a comprehensive hierarchy of custom exceptions for the MailMate
email management system. These exceptions provide detailed error information and
enable robust error handling throughout the application.

Author: MailMate Development Team
Created: October 15, 2025
"""

from typing import Any, Dict, List, Optional, Union
import traceback
from datetime import datetime


class MailMateError(Exception):
    """
    Base exception class for all MailMate-specific errors.
    
    This is the root exception class that all other MailMate exceptions inherit from.
    It provides common functionality for error tracking, logging, and debugging.
    
    Attributes:
        message (str): Human-readable error message
        error_code (str): Unique error code for programmatic handling
        details (Dict[str, Any]): Additional error context and details
        timestamp (datetime): When the error occurred
        traceback_info (str): Stack trace information for debugging
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Initialize a MailMate error.
        
        Args:
            message: Human-readable error description
            error_code: Unique identifier for this error type
            details: Additional context information
            original_exception: The original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.timestamp = datetime.now()
        self.original_exception = original_exception
        self.traceback_info = traceback.format_exc() if original_exception else None
        
    def __str__(self) -> str:
        """Return a formatted string representation of the error."""
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"details={self.details})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary for serialization.
        
        Returns:
            Dictionary containing all exception details
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback_info
        }


class ValidationError(MailMateError):
    """
    Raised when input validation fails.
    
    This exception is used when user inputs, configuration values, or data
    do not meet the required validation criteria.
    
    Examples:
        - Invalid email format
        - Missing required parameters
        - Out-of-range values
        - Invalid data types
    """
    
    def __init__(
        self, 
        message: str, 
        field_name: Optional[str] = None,
        invalid_value: Any = None,
        expected_type: Optional[type] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a validation error.
        
        Args:
            message: Description of the validation failure
            field_name: Name of the field that failed validation
            invalid_value: The value that failed validation
            expected_type: The expected data type
            constraints: Validation constraints that were violated
        """
        details = {
            'field_name': field_name,
            'invalid_value': str(invalid_value) if invalid_value is not None else None,
            'expected_type': expected_type.__name__ if expected_type else None,
            'constraints': constraints or {}
        }
        super().__init__(message, 'VALIDATION_ERROR', details)


class ExportError(MailMateError):
    """
    Raised when data export operations fail.
    
    This exception covers failures in exporting emails, analytics, or other
    data to various formats (CSV, JSON, etc.).
    
    Examples:
        - File system write errors
        - Data formatting issues
        - Export format not supported
        - Insufficient disk space
    """
    
    def __init__(
        self, 
        message: str, 
        export_type: Optional[str] = None,
        file_path: Optional[str] = None,
        record_count: Optional[int] = None
    ) -> None:
        """
        Initialize an export error.
        
        Args:
            message: Description of the export failure
            export_type: Type of export (CSV, JSON, etc.)
            file_path: Path where export was attempted
            record_count: Number of records being exported
        """
        details = {
            'export_type': export_type,
            'file_path': file_path,
            'record_count': record_count
        }
        super().__init__(message, 'EXPORT_ERROR', details)


class SessionError(MailMateError):
    """
    Raised when session management operations fail.
    
    This exception covers failures in creating, updating, retrieving, or
    managing user dashboard sessions.
    
    Examples:
        - Session not found
        - Session expired
        - Invalid session data
        - Session storage failures
    """
    
    def __init__(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """
        Initialize a session error.
        
        Args:
            message: Description of the session failure
            session_id: ID of the affected session
            user_id: ID of the user associated with the session
            operation: The session operation that failed
        """
        details = {
            'session_id': session_id,
            'user_id': user_id,
            'operation': operation
        }
        super().__init__(message, 'SESSION_ERROR', details)


class ClassificationError(MailMateError):
    """
    Raised when email classification operations fail.
    
    This exception covers failures in the email classification and
    categorization processes.
    
    Examples:
        - Model loading failures
        - Classification prediction errors
        - Invalid email content
        - Training data issues
    """
    
    def __init__(
        self, 
        message: str, 
        email_id: Optional[str] = None,
        model_type: Optional[str] = None,
        feature_count: Optional[int] = None
    ) -> None:
        """
        Initialize a classification error.
        
        Args:
            message: Description of the classification failure
            email_id: ID of the email being classified
            model_type: Type of classification model
            feature_count: Number of features extracted
        """
        details = {
            'email_id': email_id,
            'model_type': model_type,
            'feature_count': feature_count
        }
        super().__init__(message, 'CLASSIFICATION_ERROR', details)


class SummarizationError(MailMateError):
    """
    Raised when email summarization operations fail.
    
    This exception covers failures in generating summaries, extracting
    key points, or performing sentiment analysis on emails.
    
    Examples:
        - Model inference failures
        - Text processing errors
        - Content too long or too short
        - Language detection issues
    """
    
    def __init__(
        self, 
        message: str, 
        email_id: Optional[str] = None,
        text_length: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> None:
        """
        Initialize a summarization error.
        
        Args:
            message: Description of the summarization failure
            email_id: ID of the email being summarized
            text_length: Length of the text being processed
            model_name: Name of the summarization model
        """
        details = {
            'email_id': email_id,
            'text_length': text_length,
            'model_name': model_name
        }
        super().__init__(message, 'SUMMARIZATION_ERROR', details)


class TextToSpeechError(MailMateError):
    """
    Raised when text-to-speech operations fail.
    
    This exception covers failures in converting text to speech,
    audio file generation, or TTS engine issues.
    
    Examples:
        - TTS engine failures
        - Audio file write errors
        - Unsupported languages
        - Network connectivity issues (for cloud TTS)
    """
    
    def __init__(
        self, 
        message: str, 
        engine_type: Optional[str] = None,
        text_length: Optional[int] = None,
        language: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> None:
        """
        Initialize a text-to-speech error.
        
        Args:
            message: Description of the TTS failure
            engine_type: Type of TTS engine being used
            text_length: Length of text being converted
            language: Language code for TTS
            output_path: Path where audio was to be saved
        """
        details = {
            'engine_type': engine_type,
            'text_length': text_length,
            'language': language,
            'output_path': output_path
        }
        super().__init__(message, 'TTS_ERROR', details)


class EmailLoaderError(MailMateError):
    """
    Raised when email loading and processing operations fail.
    
    This exception covers failures in loading emails from various sources,
    parsing email content, or generating synthetic email data.
    
    Examples:
        - Email server connection failures
        - Email parsing errors
        - Authentication failures
        - Mailbox access issues
    """
    
    def __init__(
        self, 
        message: str, 
        source: Optional[str] = None,
        email_count: Optional[int] = None,
        connection_type: Optional[str] = None
    ) -> None:
        """
        Initialize an email loader error.
        
        Args:
            message: Description of the loading failure
            source: Source of emails (IMAP, file, synthetic, etc.)
            email_count: Number of emails being processed
            connection_type: Type of connection used
        """
        details = {
            'source': source,
            'email_count': email_count,
            'connection_type': connection_type
        }
        super().__init__(message, 'EMAIL_LOADER_ERROR', details)


class ConfigurationError(MailMateError):
    """
    Raised when configuration or initialization issues occur.
    
    This exception covers failures in loading configuration files,
    initializing modules, or setting up the application environment.
    
    Examples:
        - Missing configuration files
        - Invalid configuration values
        - Environment setup failures
        - Dependency initialization errors
    """
    
    def __init__(
        self, 
        message: str, 
        config_file: Optional[str] = None,
        missing_keys: Optional[List[str]] = None,
        module_name: Optional[str] = None
    ) -> None:
        """
        Initialize a configuration error.
        
        Args:
            message: Description of the configuration failure
            config_file: Path to the configuration file
            missing_keys: List of missing configuration keys
            module_name: Name of the module being configured
        """
        details = {
            'config_file': config_file,
            'missing_keys': missing_keys or [],
            'module_name': module_name
        }
        super().__init__(message, 'CONFIGURATION_ERROR', details)


class APIError(MailMateError):
    """
    Raised when API endpoint operations fail.
    
    This exception covers failures in REST API endpoints, request processing,
    or response generation.
    
    Examples:
        - Invalid request parameters
        - Authentication failures
        - Rate limiting violations
        - Internal server errors
    """
    
    def __init__(
        self, 
        message: str, 
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize an API error.
        
        Args:
            message: Description of the API failure
            endpoint: API endpoint that failed
            method: HTTP method used
            status_code: HTTP status code
            request_data: Request data that caused the error
        """
        details = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'request_data': request_data
        }
        super().__init__(message, 'API_ERROR', details)


# Utility functions for error handling

def validate_required_params(params: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that all required parameters are present and not None.
    
    Args:
        params: Dictionary of parameters to validate
        required_keys: List of required parameter keys
        
    Raises:
        ValidationError: If any required parameter is missing or None
    """
    missing_keys = []
    for key in required_keys:
        if key not in params or params[key] is None:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValidationError(
            f"Missing required parameters: {', '.join(missing_keys)}",
            constraints={'required_keys': required_keys, 'missing_keys': missing_keys}
        )


def validate_type(value: Any, expected_type: type, field_name: str) -> None:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        field_name: Name of the field being validated
        
    Raises:
        ValidationError: If the value is not of the expected type
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
            field_name=field_name,
            invalid_value=value,
            expected_type=expected_type
        )


def validate_string_length(
    value: str, 
    field_name: str, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None
) -> None:
    """
    Validate string length constraints.
    
    Args:
        value: String to validate
        field_name: Name of the field being validated
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Raises:
        ValidationError: If string length is outside the allowed range
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"Field '{field_name}' must be a string",
            field_name=field_name,
            invalid_value=value,
            expected_type=str
        )
    
    length = len(value)
    constraints = {}
    
    if min_length is not None:
        constraints['min_length'] = min_length
        if length < min_length:
            raise ValidationError(
                f"Field '{field_name}' must be at least {min_length} characters long, got {length}",
                field_name=field_name,
                invalid_value=value,
                constraints=constraints
            )
    
    if max_length is not None:
        constraints['max_length'] = max_length
        if length > max_length:
            raise ValidationError(
                f"Field '{field_name}' must be at most {max_length} characters long, got {length}",
                field_name=field_name,
                invalid_value=value,
                constraints=constraints
            )


def validate_choice(value: Any, field_name: str, valid_choices: List[Any]) -> None:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: Value to validate
        field_name: Name of the field being validated
        valid_choices: List of valid choices
        
    Raises:
        ValidationError: If the value is not in the list of valid choices
    """
    if value not in valid_choices:
        raise ValidationError(
            f"Field '{field_name}' must be one of {valid_choices}, got '{value}'",
            field_name=field_name,
            invalid_value=value,
            constraints={'valid_choices': valid_choices}
        )


def validate_string(
    value: Any, 
    field_name: str, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None,
    required: bool = True
) -> None:
    """
    Validate string field with comprehensive checks.
    
    Args:
        value: Value to validate
        field_name: Name of the field being validated
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        required: Whether the field is required
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(
                f"Field '{field_name}' is required",
                field_name=field_name,
                invalid_value=value
            )
        return
    
    if not isinstance(value, str):
        raise ValidationError(
            f"Field '{field_name}' must be a string, got {type(value).__name__}",
            field_name=field_name,
            invalid_value=value,
            expected_type=str
        )
    
    length = len(value)
    constraints = {}
    
    if min_length is not None:
        constraints['min_length'] = min_length
        if length < min_length:
            raise ValidationError(
                f"Field '{field_name}' must be at least {min_length} characters long, got {length}",
                field_name=field_name,
                invalid_value=value,
                constraints=constraints
            )
    
    if max_length is not None:
        constraints['max_length'] = max_length
        if length > max_length:
            raise ValidationError(
                f"Field '{field_name}' must be at most {max_length} characters long, got {length}",
                field_name=field_name,
                invalid_value=value,
                constraints=constraints
            )


def validate_dict(value: Any, field_name: str, required: bool = True) -> None:
    """
    Validate dictionary field.
    
    Args:
        value: Value to validate
        field_name: Name of the field being validated
        required: Whether the field is required
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(
                f"Field '{field_name}' is required",
                field_name=field_name,
                invalid_value=value
            )
        return
    
    if not isinstance(value, dict):
        raise ValidationError(
            f"Field '{field_name}' must be a dictionary, got {type(value).__name__}",
            field_name=field_name,
            invalid_value=value,
            expected_type=dict
        )


def validate_list(value: Any, field_name: str, required: bool = True) -> None:
    """
    Validate list field.
    
    Args:
        value: Value to validate
        field_name: Name of the field being validated
        required: Whether the field is required
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(
                f"Field '{field_name}' is required",
                field_name=field_name,
                invalid_value=value
            )
        return
    
    if not isinstance(value, list):
        raise ValidationError(
            f"Field '{field_name}' must be a list, got {type(value).__name__}",
            field_name=field_name,
            invalid_value=value,
            expected_type=list
        )


def handle_exception(
    exception: Exception, 
    operation: str, 
    logger=None,
    reraise_as: Optional[type] = None
) -> None:
    """
    Handle exceptions with consistent logging and optional re-raising.
    
    Args:
        exception: The exception that occurred
        operation: Description of the operation that failed
        logger: Logger instance for error logging
        reraise_as: Optional exception class to re-raise as
        
    Raises:
        The original exception or a new exception of type reraise_as
    """
    error_msg = f"Error during {operation}: {str(exception)}"
    
    if logger:
        logger.error(error_msg, exc_info=True)
    
    if reraise_as:
        if issubclass(reraise_as, MailMateError):
            raise reraise_as(error_msg, original_exception=exception)
        else:
            raise reraise_as(error_msg) from exception
    else:
        raise


# Exception registry for programmatic access
EXCEPTION_REGISTRY = {
    'VALIDATION_ERROR': ValidationError,
    'EXPORT_ERROR': ExportError,
    'SESSION_ERROR': SessionError,
    'CLASSIFICATION_ERROR': ClassificationError,
    'SUMMARIZATION_ERROR': SummarizationError,
    'TTS_ERROR': TextToSpeechError,
    'EMAIL_LOADER_ERROR': EmailLoaderError,
    'CONFIGURATION_ERROR': ConfigurationError,
    'API_ERROR': APIError
}


def get_exception_class(error_code: str) -> type:
    """
    Get exception class by error code.
    
    Args:
        error_code: Error code string
        
    Returns:
        Exception class corresponding to the error code
        
    Raises:
        KeyError: If error code is not found in registry
    """
    return EXCEPTION_REGISTRY.get(error_code, MailMateError)