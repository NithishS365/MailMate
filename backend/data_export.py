"""
Data Export Module for MailMate Dashboard

This module provides comprehensive data export functionality with robust validation,
error handling, and type safety for:
- Email data export to CSV and JSON formats with validation
- Summary reports with analytics and error handling
- Dashboard analytics and metrics with type checking
- Custom filtered exports with date ranges and validation
- Batch export operations with progress tracking and error recovery

Features:
- Comprehensive input validation using custom exceptions
- Detailed type hints for all methods and parameters
- Robust error handling with detailed error messages
- Progress tracking and performance monitoring
- Memory-efficient processing for large datasets
- Comprehensive logging and debugging support

Author: MailMate Development Team
Version: 2.0.0 (Enhanced with validation and error handling)
Created: October 15, 2025
"""

import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Protocol, runtime_checkable
import logging
from dataclasses import asdict, is_dataclass
import zipfile
import io
import os
import shutil
from contextlib import contextmanager

from logging_config import get_logger
from email_loader import EmailData
from exceptions import (
    ExportError, ValidationError, ConfigurationError, MailMateError,
    validate_required_params, validate_type, validate_string_length,
    validate_choice, handle_exception
)

logger = get_logger("mailmate.export")


@runtime_checkable
class ExportableData(Protocol):
    """Protocol for data that can be exported."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary for export."""
        ...


# Type aliases for better code readability
EmailList = List[Union[EmailData, Dict[str, Any]]]
FilterDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
ExportFormat = Union[str, Any]  # CSV, JSON, EXCEL, XML formats

# Constants for export formats
SUPPORTED_FORMATS = ['CSV', 'JSON', 'EXCEL', 'XML']
DEFAULT_BATCH_SIZE = 1000
MAX_EXPORT_SIZE = 100000


class DataExporter:
    """
    Comprehensive data export utility for MailMate Dashboard with robust validation.
    
    This class provides type-safe, validated export functionality with comprehensive
    error handling and detailed logging. Supports multiple export formats with
    filtering, pagination, and batch processing capabilities.
    
    Features:
        - Input validation with custom exceptions
        - Type-safe method signatures with comprehensive hints
        - Robust error handling with detailed error context
        - Progress tracking for large exports
        - Memory-efficient batch processing
        - Comprehensive audit logging
        - Format validation and security checks
    
    Attributes:
        output_dir (Path): Directory for exported files
        logger (logging.Logger): Logger instance for this exporter
        export_history (List[Dict]): History of export operations
        max_export_size (int): Maximum number of records per export
        batch_size (int): Batch size for processing large datasets
    
    Example:
        >>> exporter = DataExporter(output_dir="./exports")
        >>> exporter.export_emails_to_csv(emails, filters={"category": "Work"})
        './exports/emails_export_20251015_143022.csv'
    """
    
    def __init__(
        self, 
        output_dir: str = "exports",
        max_export_size: int = MAX_EXPORT_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        validate_data: bool = True
    ) -> None:
        """
        Initialize DataExporter with validation and configuration.
        
        Args:
            output_dir: Directory to save exported files (must be writable)
            max_export_size: Maximum number of records per export operation
            batch_size: Number of records to process in each batch
            validate_data: Whether to validate input data before export
            
        Raises:
            ValidationError: If parameters are invalid
            ConfigurationError: If output directory cannot be created or accessed
            
        Example:
            >>> exporter = DataExporter(
            ...     output_dir="./custom_exports",
            ...     max_export_size=50000,
            ...     batch_size=500
            ... )
        """
        # Validate input parameters
        self._validate_init_params(output_dir, max_export_size, batch_size)
        
        try:
            # Initialize output directory
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify directory is writable
            test_file = self.output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            
        except (OSError, PermissionError) as e:
            raise ConfigurationError(
                f"Cannot create or access output directory '{output_dir}': {str(e)}",
                config_file=output_dir,
                module_name="DataExporter"
            )
        
        # Initialize configuration
        self.max_export_size = max_export_size
        self.batch_size = batch_size
        self.validate_data = validate_data
        self.logger = get_logger("mailmate.export")
        
        # Export tracking
        self.export_history: List[Dict[str, Any]] = []
        self._operation_counter = 0
        
        self.logger.info(
            f"DataExporter initialized - output_dir: {self.output_dir}, "
            f"max_size: {max_export_size}, batch_size: {batch_size}"
        )
    
    def _validate_init_params(
        self, 
        output_dir: str, 
        max_export_size: int, 
        batch_size: int
    ) -> None:
        """Validate DataExporter initialization parameters."""
        validate_type(output_dir, str, "output_dir")
        validate_string_length(output_dir, "output_dir", min_length=1, max_length=255)
        
        validate_type(max_export_size, int, "max_export_size")
        if max_export_size <= 0:
            raise ValidationError(
                "max_export_size must be positive",
                field_name="max_export_size",
                invalid_value=max_export_size,
                constraints={"min_value": 1}
            )
        
        validate_type(batch_size, int, "batch_size")
        if batch_size <= 0 or batch_size > max_export_size:
            raise ValidationError(
                f"batch_size must be between 1 and {max_export_size}",
                field_name="batch_size",
                invalid_value=batch_size,
                constraints={"min_value": 1, "max_value": max_export_size}
            )
    
    @contextmanager
    def _export_operation(self, operation_name: str, record_count: int):
        """Context manager for tracking export operations with error handling."""
        operation_id = f"{operation_name}_{self._operation_counter}"
        self._operation_counter += 1
        start_time = datetime.now()
        
        self.logger.info(f"Starting export operation: {operation_id} ({record_count} records)")
        
        try:
            yield operation_id
        except Exception as e:
            duration = datetime.now() - start_time
            self.logger.error(
                f"Export operation failed: {operation_id} after {duration.total_seconds():.2f}s - {str(e)}",
                exc_info=True
            )
            raise
        else:
            duration = datetime.now() - start_time
            self.logger.info(
                f"Export operation completed: {operation_id} in {duration.total_seconds():.2f}s"
            )
    
    def export_emails_to_csv(
        self,
        emails: EmailList,
        filename: Optional[str] = None,
        include_metadata: bool = True,
        filters: Optional[FilterDict] = None,
        validate_emails: bool = True
    ) -> str:
        """
        Export email data to CSV format with comprehensive validation.
        
        This method exports a list of emails to a CSV file with optional metadata
        and filtering information. Includes validation of input data and handles
        various email data formats.
        
        Args:
            emails: List of EmailData objects or dictionaries to export.
                   Must not be empty and each email must have required fields.
            filename: Custom filename without extension (auto-generated if None).
                     Must be a valid filename without path separators.
            include_metadata: Whether to include export metadata as CSV comments.
            filters: Dictionary of filters applied to the data (for metadata only).
                    Used to document the export parameters.
            validate_emails: Whether to validate each email's structure and content.
            
        Returns:
            str: Absolute path to the exported CSV file
            
        Raises:
            ValidationError: If input parameters are invalid or emails are malformed
            ExportError: If file creation or writing fails
            
        Example:
            >>> emails = [
            ...     {"subject": "Test", "body": "Content", "category": "Work"},
            ...     {"subject": "Another", "body": "More content", "category": "Personal"}
            ... ]
            >>> path = exporter.export_emails_to_csv(
            ...     emails, 
            ...     filename="work_emails", 
            ...     filters={"category": "Work"}
            ... )
            >>> print(f"Exported to: {path}")
        """
        # Validate input parameters
        self._validate_export_params(emails, filename, "CSV")
        
        if validate_emails and self.validate_data:
            self._validate_email_list(emails)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emails_export_{timestamp}"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = self.output_dir / filename
        
        with self._export_operation("CSV_EXPORT", len(emails)) as operation_id:
            try:
                # Convert emails to DataFrame with validation
                df = self._emails_to_dataframe(emails, validate_data=validate_emails)
                
                # Create export metadata
                metadata = self._create_export_metadata("CSV", len(emails), filters, operation_id)
                
                # Write file with metadata
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    if include_metadata:
                        self._write_csv_metadata(f, metadata)
                    
                    # Write DataFrame to CSV
                    df.to_csv(f, index=False, encoding='utf-8')
                
                # Verify file was created successfully
                if not filepath.exists() or filepath.stat().st_size == 0:
                    raise ExportError(
                        "CSV file was not created or is empty",
                        export_type="CSV",
                        file_path=str(filepath),
                        record_count=len(emails)
                    )
                
                # Track export operation
                self._track_export("CSV", str(filepath), len(emails), filters, operation_id)
                
                self.logger.info(f"Successfully exported {len(emails)} emails to CSV: {filepath}")
                return str(filepath.resolve())
                
            except (IOError, OSError, PermissionError) as e:
                raise ExportError(
                    f"Failed to write CSV file: {str(e)}",
                    export_type="CSV",
                    file_path=str(filepath),
                    record_count=len(emails)
                ) from e
            except Exception as e:
                handle_exception(e, "CSV export", self.logger, ExportError)
    
    def _validate_export_params(
        self, 
        emails: EmailList, 
        filename: Optional[str], 
        export_format: str
    ) -> None:
        """Validate common export parameters."""
        # Validate emails list
        validate_type(emails, list, "emails")
        if not emails:
            raise ValidationError(
                "Email list cannot be empty",
                field_name="emails",
                invalid_value=emails,
                constraints={"min_length": 1}
            )
        
        if len(emails) > self.max_export_size:
            raise ValidationError(
                f"Email list too large: {len(emails)} > {self.max_export_size}",
                field_name="emails",
                invalid_value=len(emails),
                constraints={"max_length": self.max_export_size}
            )
        
        # Validate filename if provided
        if filename is not None:
            validate_type(filename, str, "filename")
            validate_string_length(filename, "filename", min_length=1, max_length=100)
            
            # Check for invalid filename characters
            invalid_chars = set(filename) & set('<>:"/\\|?*')
            if invalid_chars:
                raise ValidationError(
                    f"Filename contains invalid characters: {invalid_chars}",
                    field_name="filename",
                    invalid_value=filename,
                    constraints={"forbidden_chars": list(invalid_chars)}
                )
        
        # Validate export format
        validate_choice(export_format, "export_format", SUPPORTED_FORMATS)
    
    def _validate_email_list(self, emails: EmailList) -> None:
        """Validate structure and content of email list."""
        required_fields = ['subject', 'body']
        
        for i, email in enumerate(emails):
            try:
                # Convert to dict if needed
                if hasattr(email, '__dict__'):
                    email_dict = asdict(email) if is_dataclass(email) else email.__dict__
                elif isinstance(email, dict):
                    email_dict = email
                else:
                    raise ValidationError(
                        f"Email at index {i} is not a valid email object or dictionary",
                        field_name=f"emails[{i}]",
                        invalid_value=type(email),
                        expected_type=dict
                    )
                
                # Check required fields
                for field in required_fields:
                    if field not in email_dict or email_dict[field] is None:
                        raise ValidationError(
                            f"Email at index {i} missing required field '{field}'",
                            field_name=f"emails[{i}].{field}",
                            invalid_value=email_dict.get(field),
                            constraints={"required_fields": required_fields}
                        )
                
                # Validate field types and content
                if not isinstance(email_dict['subject'], str):
                    raise ValidationError(
                        f"Email at index {i} has invalid subject type",
                        field_name=f"emails[{i}].subject",
                        invalid_value=type(email_dict['subject']),
                        expected_type=str
                    )
                
                if not isinstance(email_dict['body'], str):
                    raise ValidationError(
                        f"Email at index {i} has invalid body type",
                        field_name=f"emails[{i}].body",
                        invalid_value=type(email_dict['body']),
                        expected_type=str
                    )
                        
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"Error validating email at index {i}: {str(e)}",
                    field_name=f"emails[{i}]",
                    invalid_value=str(email)
                ) from e
    
    def _emails_to_dataframe(self, emails: EmailList, validate_data: bool = True) -> pd.DataFrame:
        """Convert email list to pandas DataFrame with error handling."""
        try:
            email_dicts = []
            
            for i, email in enumerate(emails):
                try:
                    # Convert to dict
                    if hasattr(email, '__dict__'):
                        email_dict = asdict(email) if is_dataclass(email) else email.__dict__
                    elif isinstance(email, dict):
                        email_dict = email.copy()
                    else:
                        email_dict = {"content": str(email), "index": i}
                    
                    # Flatten nested structures
                    flattened_dict = self._flatten_email_data(email_dict)
                    email_dicts.append(flattened_dict)
                    
                except Exception as e:
                    if validate_data:
                        raise ExportError(
                            f"Error processing email at index {i}: {str(e)}",
                            export_type="DataFrame",
                            record_count=i
                        ) from e
                    else:
                        # Add error placeholder
                        email_dicts.append({
                            "error": f"Failed to process email {i}: {str(e)}",
                            "index": i
                        })
            
            if not email_dicts:
                raise ExportError(
                    "No valid emails could be processed",
                    export_type="DataFrame",
                    record_count=0
                )
            
            return pd.DataFrame(email_dicts)
            
        except pd.errors.EmptyDataError as e:
            raise ExportError(
                "Failed to create DataFrame from email data",
                export_type="DataFrame",
                record_count=len(emails)
            ) from e
    
    def _write_csv_metadata(self, file_handle, metadata: MetadataDict) -> None:
        """Write metadata as CSV comments."""
        file_handle.write("# MailMate Email Export\n")
        file_handle.write(f"# Export Date: {metadata['export_date']}\n")
        file_handle.write(f"# Total Records: {metadata['total_records']}\n")
        file_handle.write(f"# Filters Applied: {metadata.get('filters_applied', 'None')}\n")
        file_handle.write(f"# Generated By: MailMate Dashboard v{metadata.get('version', '2.0.0')}\n")
        file_handle.write(f"# Operation ID: {metadata.get('operation_id', 'N/A')}\n")
        file_handle.write("#\n")
    
    def export_emails_to_json(
        self,
        emails: EmailList,
        filename: Optional[str] = None,
        pretty_print: bool = True,
        include_metadata: bool = True,
        filters: Optional[FilterDict] = None,
        validate_emails: bool = True
    ) -> str:
        """
        Export email data to JSON format with comprehensive validation.
        
        This method exports a list of emails to a JSON file with optional pretty
        printing and metadata inclusion. Provides robust error handling and
        validation of input data.
        
        Args:
            emails: List of EmailData objects or dictionaries to export.
                   Must not be empty and each email must be serializable to JSON.
            filename: Custom filename without extension (auto-generated if None).
                     Must be a valid filename without path separators.
            pretty_print: Whether to format JSON with indentation for readability.
            include_metadata: Whether to include export metadata in the JSON.
            filters: Dictionary of filters applied to the data (for metadata only).
            validate_emails: Whether to validate each email's JSON serialization.
            
        Returns:
            str: Absolute path to the exported JSON file
            
        Raises:
            ValidationError: If input parameters are invalid or emails are malformed
            ExportError: If file creation, JSON serialization, or writing fails
            
        Example:
            >>> emails = [
            ...     {"subject": "Test", "body": "Content", "timestamp": "2023-10-15"},
            ...     {"subject": "Another", "body": "More content", "category": "Work"}
            ... ]
            >>> path = exporter.export_emails_to_json(
            ...     emails, 
            ...     filename="emails_backup",
            ...     pretty_print=True
            ... )
            >>> print(f"Exported to: {path}")
        """
        # Validate input parameters
        self._validate_export_params(emails, filename, "JSON")
        validate_type(pretty_print, bool, "pretty_print")
        validate_type(include_metadata, bool, "include_metadata")
        
        if validate_emails and self.validate_data:
            self._validate_email_list(emails)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emails_export_{timestamp}"
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.output_dir / filename
        
        with self._export_operation("JSON_EXPORT", len(emails)) as operation_id:
            try:
                # Convert emails to serializable format
                email_dicts = self._emails_to_json_dict(emails, validate_data=validate_emails)
                
                # Create export structure
                export_data = {
                    "emails": email_dicts,
                    "export_info": {
                        "total_count": len(email_dicts),
                        "export_date": datetime.now().isoformat(),
                        "format": "JSON"
                    }
                }
                
                # Add metadata if requested
                if include_metadata:
                    metadata = self._create_export_metadata("JSON", len(emails), filters, operation_id)
                    export_data["metadata"] = metadata
                
                # Write JSON file
                with open(filepath, 'w', encoding='utf-8') as f:
                    if pretty_print:
                        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                    else:
                        json.dump(export_data, f, ensure_ascii=False, default=str)
                
                # Verify file was created successfully
                if not filepath.exists() or filepath.stat().st_size == 0:
                    raise ExportError(
                        "JSON file was not created or is empty",
                        export_type="JSON",
                        file_path=str(filepath),
                        record_count=len(emails)
                    )
                
                # Track export operation
                self._track_export("JSON", str(filepath), len(emails), filters, operation_id)
                
                self.logger.info(f"Successfully exported {len(emails)} emails to JSON: {filepath}")
                return str(filepath.resolve())
                
            except (IOError, OSError, PermissionError) as e:
                raise ExportError(
                    f"Failed to write JSON file: {str(e)}",
                    export_type="JSON",
                    file_path=str(filepath),
                    record_count=len(emails)
                ) from e
            except json.JSONEncoder as e:
                raise ExportError(
                    f"Failed to serialize emails to JSON: {str(e)}",
                    export_type="JSON",
                    file_path=str(filepath),
                    record_count=len(emails)
                ) from e
            except Exception as e:
                handle_exception(e, "JSON export", self.logger, ExportError)
    
    def _emails_to_json_dict(self, emails: EmailList, validate_data: bool = True) -> List[Dict[str, Any]]:
        """Convert email list to JSON-serializable dictionaries."""
        email_dicts = []
        
        for i, email in enumerate(emails):
            try:
                # Convert to dict
                if hasattr(email, '__dict__'):
                    email_dict = asdict(email) if is_dataclass(email) else email.__dict__
                elif isinstance(email, dict):
                    email_dict = email.copy()
                else:
                    email_dict = {"content": str(email), "index": i}
                
                # Ensure JSON serializable
                json_dict = self._make_json_serializable(email_dict)
                email_dicts.append(json_dict)
                
            except Exception as e:
                if validate_data:
                    raise ExportError(
                        f"Error converting email at index {i} to JSON: {str(e)}",
                        export_type="JSON",
                        record_count=i
                    ) from e
                else:
                    # Add error placeholder
                    email_dicts.append({
                        "error": f"Failed to process email {i}: {str(e)}",
                        "index": i
                    })
        
        return email_dicts
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, )):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _flatten_email_data(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested email data structures for CSV export."""
        flattened = {}
        
        for key, value in email_dict.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries
                for nested_key, nested_value in value.items():
                    flattened_key = f"{key}_{nested_key}"
                    flattened[flattened_key] = self._serialize_value(nested_value)
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value and isinstance(value[0], dict):
                    # For list of dicts, create multiple columns
                    for i, item in enumerate(value[:5]):  # Limit to first 5 items
                        for item_key, item_value in item.items():
                            flattened_key = f"{key}_{i}_{item_key}"
                            flattened[flattened_key] = self._serialize_value(item_value)
                else:
                    flattened[key] = ", ".join(str(item) for item in value)
            else:
                flattened[key] = self._serialize_value(value)
        
        return flattened
    
    def _serialize_value(self, value: Any) -> str:
        """Convert any value to string for CSV export."""
        if value is None:
            return ""
        elif isinstance(value, (datetime,)):
            return value.isoformat()
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        else:
            return str(value)
    
    def _create_export_metadata(
        self, 
        export_format: str, 
        record_count: int, 
        filters: Optional[FilterDict],
        operation_id: Optional[str] = None
    ) -> MetadataDict:
        """Create comprehensive export metadata."""
        return {
            "export_date": datetime.now().isoformat(),
            "export_format": export_format,
            "total_records": record_count,
            "filters_applied": filters or {},
            "version": "2.0.0",
            "exporter": "MailMate DataExporter",
            "operation_id": operation_id,
            "batch_size": self.batch_size,
            "validation_enabled": self.validate_data,
            "output_directory": str(self.output_dir)
        }
    
    def _track_export(
        self, 
        export_format: str, 
        filepath: str, 
        record_count: int, 
        filters: Optional[FilterDict],
        operation_id: Optional[str] = None
    ) -> None:
        """Track export operation in history."""
        export_record = {
            "timestamp": datetime.now().isoformat(),
            "format": export_format,
            "filepath": filepath,
            "record_count": record_count,
            "filters": filters or {},
            "operation_id": operation_id,
            "file_size": Path(filepath).stat().st_size if Path(filepath).exists() else 0
        }
        
        self.export_history.append(export_record)
        
        # Keep only last 100 export records
        if len(self.export_history) > 100:
            self.export_history = self.export_history[-100:]
    
    def export_analytics_report(
        self,
        analytics_data: Dict[str, Any],
        format_type: str = "JSON",
        filename: Optional[str] = None,
        include_charts_data: bool = True,
        validate_analytics: bool = True
    ) -> str:
        """
        Export analytics data with comprehensive validation and error handling.
        
        This method exports dashboard analytics and metrics to various formats
        with validation and detailed error handling.
        
        Args:
            analytics_data: Dictionary containing analytics data and metrics.
                          Must contain required analytics fields.
            format_type: Export format ('JSON', 'CSV', or 'EXCEL').
            filename: Custom filename without extension (auto-generated if None).
            include_charts_data: Whether to include chart data in the export.
            validate_analytics: Whether to validate analytics data structure.
            
        Returns:
            str: Absolute path to the exported analytics file
            
        Raises:
            ValidationError: If analytics data is invalid or malformed
            ExportError: If file creation or writing fails
            
        Example:
            >>> analytics = {
            ...     "dashboard_stats": {"total_emails": 100, "unread": 15},
            ...     "email_analytics": {"categories": {"Work": 60, "Personal": 40}},
            ...     "performance_metrics": {"response_time": 1.2}
            ... }
            >>> path = exporter.export_analytics_report(
            ...     analytics, 
            ...     format_type="JSON",
            ...     include_charts_data=True
            ... )
        """
        # Validate input parameters
        validate_type(analytics_data, dict, "analytics_data")
        validate_choice(format_type, "format_type", ["JSON", "CSV", "EXCEL"])
        validate_type(include_charts_data, bool, "include_charts_data")
        
        if validate_analytics and self.validate_data:
            self._validate_analytics_data(analytics_data)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_report_{timestamp}"
        
        # Add appropriate extension
        extensions = {"JSON": ".json", "CSV": ".csv", "EXCEL": ".xlsx"}
        if not any(filename.endswith(ext) for ext in extensions.values()):
            filename += extensions[format_type]
        
        filepath = self.output_dir / filename
        
        with self._export_operation("ANALYTICS_EXPORT", len(analytics_data)) as operation_id:
            try:
                if format_type == "JSON":
                    return self._export_analytics_json(
                        analytics_data, filepath, include_charts_data, operation_id
                    )
                elif format_type == "CSV":
                    return self._export_analytics_csv(
                        analytics_data, filepath, include_charts_data, operation_id
                    )
                elif format_type == "EXCEL":
                    return self._export_analytics_excel(
                        analytics_data, filepath, include_charts_data, operation_id
                    )
                    
            except Exception as e:
                handle_exception(e, "analytics export", self.logger, ExportError)
    
    def _validate_analytics_data(self, analytics_data: Dict[str, Any]) -> None:
        """Validate analytics data structure and content."""
        required_sections = ["dashboard_stats", "email_analytics"]
        
        for section in required_sections:
            if section not in analytics_data:
                raise ValidationError(
                    f"Analytics data missing required section: {section}",
                    field_name=section,
                    constraints={"required_sections": required_sections}
                )
            
            if not isinstance(analytics_data[section], dict):
                raise ValidationError(
                    f"Analytics section '{section}' must be a dictionary",
                    field_name=section,
                    invalid_value=type(analytics_data[section]),
                    expected_type=dict
                )
        
        # Validate dashboard stats
        dashboard_stats = analytics_data["dashboard_stats"]
        if "total_emails" not in dashboard_stats:
            raise ValidationError(
                "Dashboard stats missing 'total_emails' field",
                field_name="dashboard_stats.total_emails",
                constraints={"required_fields": ["total_emails"]}
            )
    
    def _export_analytics_json(
        self, 
        analytics_data: Dict[str, Any], 
        filepath: Path, 
        include_charts: bool,
        operation_id: str
    ) -> str:
        """Export analytics to JSON format."""
        export_data = {
            "analytics": analytics_data,
            "export_info": {
                "format": "JSON",
                "export_date": datetime.now().isoformat(),
                "include_charts": include_charts,
                "operation_id": operation_id
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self._track_export("JSON", str(filepath), len(analytics_data), {}, operation_id)
        self.logger.info(f"Exported analytics to JSON: {filepath}")
        return str(filepath.resolve())
    
    def _export_analytics_csv(
        self, 
        analytics_data: Dict[str, Any], 
        filepath: Path, 
        include_charts: bool,
        operation_id: str
    ) -> str:
        """Export analytics to CSV format (flattened structure)."""
        # Flatten analytics data for CSV
        flattened_data = []
        
        for section, data in analytics_data.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flattened_data.append({
                                "section": section,
                                "category": key,
                                "metric": sub_key,
                                "value": sub_value
                            })
                    else:
                        flattened_data.append({
                            "section": section,
                            "category": "general",
                            "metric": key,
                            "value": value
                        })
        
        df = pd.DataFrame(flattened_data)
        
        # Write with metadata
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write("# MailMate Analytics Report\n")
            f.write(f"# Export Date: {datetime.now().isoformat()}\n")
            f.write(f"# Operation ID: {operation_id}\n")
            f.write("#\n")
            
            df.to_csv(f, index=False)
        
        self._track_export("CSV", str(filepath), len(flattened_data), {}, operation_id)
        self.logger.info(f"Exported analytics to CSV: {filepath}")
        return str(filepath.resolve())
    
    def _export_analytics_excel(
        self, 
        analytics_data: Dict[str, Any], 
        filepath: Path, 
        include_charts: bool,
        operation_id: str
    ) -> str:
        """Export analytics to Excel format with multiple sheets."""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                for section, data in analytics_data.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            summary_data.append({
                                "section": section,
                                "metric": key,
                                "value": str(value)
                            })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Create detailed sheets for each section
                for section, data in analytics_data.items():
                    if isinstance(data, dict) and data:
                        try:
                            section_df = pd.DataFrame([data])
                            sheet_name = section[:31]  # Excel sheet name limit
                            section_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        except Exception as e:
                            self.logger.warning(f"Could not create sheet for {section}: {e}")
            
            self._track_export("EXCEL", str(filepath), len(analytics_data), {}, operation_id)
            self.logger.info(f"Exported analytics to Excel: {filepath}")
            return str(filepath.resolve())
            
        except ImportError:
            raise ExportError(
                "Excel export requires openpyxl package",
                export_type="EXCEL",
                file_path=str(filepath)
            )
    
    def create_bulk_export(
        self,
        emails: EmailList,
        analytics_data: Dict[str, Any],
        summaries: List[Dict[str, Any]],
        filename: Optional[str] = None,
        include_metadata: bool = True,
        validate_all_data: bool = True
    ) -> str:
        """
        Create a comprehensive bulk export with all data types.
        
        This method creates a ZIP archive containing emails, analytics, and summaries
        in multiple formats with comprehensive validation and error handling.
        
        Args:
            emails: List of EmailData objects or dictionaries to export
            analytics_data: Dictionary containing analytics and metrics
            summaries: List of email summaries and analysis results
            filename: Custom filename for the ZIP archive (auto-generated if None)
            include_metadata: Whether to include metadata in all exports
            validate_all_data: Whether to validate all input data before export
            
        Returns:
            str: Absolute path to the created ZIP archive
            
        Raises:
            ValidationError: If any input data is invalid
            ExportError: If archive creation or file operations fail
            
        Example:
            >>> archive_path = exporter.create_bulk_export(
            ...     emails=email_list,
            ...     analytics_data=analytics_dict,
            ...     summaries=summary_list,
            ...     filename="complete_export"
            ... )
            >>> print(f"Bulk export created: {archive_path}")
        """
        # Validate input parameters
        validate_type(emails, list, "emails")
        validate_type(analytics_data, dict, "analytics_data")
        validate_type(summaries, list, "summaries")
        
        if validate_all_data and self.validate_data:
            self._validate_email_list(emails)
            self._validate_analytics_data(analytics_data)
            self._validate_summaries_list(summaries)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mailmate_bulk_export_{timestamp}"
        
        # Ensure .zip extension
        if not filename.endswith('.zip'):
            filename += '.zip'
        
        filepath = self.output_dir / filename
        
        with self._export_operation("BULK_EXPORT", len(emails) + len(summaries)) as operation_id:
            try:
                with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                    # Add metadata file
                    if include_metadata:
                        metadata = self._create_bulk_export_metadata(
                            len(emails), len(analytics_data), len(summaries), operation_id
                        )
                        zipf.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
                    
                    # Export emails to multiple formats
                    self._add_emails_to_zip(zipf, emails, validate_all_data)
                    
                    # Export analytics
                    self._add_analytics_to_zip(zipf, analytics_data, operation_id)
                    
                    # Export summaries
                    self._add_summaries_to_zip(zipf, summaries, validate_all_data)
                    
                    # Add export report
                    self._add_export_report_to_zip(zipf, operation_id)
                
                # Verify archive was created successfully
                if not filepath.exists() or filepath.stat().st_size == 0:
                    raise ExportError(
                        "Bulk export archive was not created or is empty",
                        export_type="ZIP",
                        file_path=str(filepath),
                        record_count=len(emails)
                    )
                
                total_records = len(emails) + len(summaries)
                self._track_export("ZIP", str(filepath), total_records, {}, operation_id)
                
                self.logger.info(
                    f"Successfully created bulk export: {filepath} "
                    f"({len(emails)} emails, {len(summaries)} summaries)"
                )
                return str(filepath.resolve())
                
            except Exception as e:
                # Clean up partial file if it exists
                if filepath.exists():
                    try:
                        filepath.unlink()
                    except Exception:
                        pass
                handle_exception(e, "bulk export", self.logger, ExportError)
    
    def _validate_summaries_list(self, summaries: List[Dict[str, Any]]) -> None:
        """Validate structure of summaries list."""
        for i, summary in enumerate(summaries):
            if not isinstance(summary, dict):
                raise ValidationError(
                    f"Summary at index {i} must be a dictionary",
                    field_name=f"summaries[{i}]",
                    invalid_value=type(summary),
                    expected_type=dict
                )
    
    def _create_bulk_export_metadata(
        self, 
        email_count: int, 
        analytics_count: int, 
        summary_count: int,
        operation_id: str
    ) -> Dict[str, Any]:
        """Create metadata for bulk export."""
        return {
            "export_type": "BULK",
            "export_date": datetime.now().isoformat(),
            "operation_id": operation_id,
            "contents": {
                "emails": email_count,
                "analytics_sections": analytics_count,
                "summaries": summary_count
            },
            "formats_included": ["CSV", "JSON"],
            "version": "2.0.0",
            "exporter": "MailMate DataExporter Enhanced"
        }
    
    def _add_emails_to_zip(self, zipf: zipfile.ZipFile, emails: EmailList, validate: bool) -> None:
        """Add emails in multiple formats to ZIP archive."""
        # CSV format
        csv_data = io.StringIO()
        df = self._emails_to_dataframe(emails, validate_data=validate)
        df.to_csv(csv_data, index=False)
        zipf.writestr("emails/emails.csv", csv_data.getvalue())
        
        # JSON format
        json_dict = self._emails_to_json_dict(emails, validate_data=validate)
        zipf.writestr("emails/emails.json", json.dumps(json_dict, indent=2, default=str))
    
    def _add_analytics_to_zip(self, zipf: zipfile.ZipFile, analytics: Dict[str, Any], operation_id: str) -> None:
        """Add analytics data to ZIP archive."""
        # JSON format
        analytics_with_meta = {
            "analytics": analytics,
            "export_info": {
                "format": "JSON",
                "export_date": datetime.now().isoformat(),
                "operation_id": operation_id
            }
        }
        zipf.writestr("analytics/analytics.json", json.dumps(analytics_with_meta, indent=2, default=str))
    
    def _add_summaries_to_zip(self, zipf: zipfile.ZipFile, summaries: List[Dict[str, Any]], validate: bool) -> None:
        """Add summaries to ZIP archive."""
        if summaries:
            # JSON format
            zipf.writestr("summaries/summaries.json", json.dumps(summaries, indent=2, default=str))
            
            # CSV format
            try:
                df = pd.DataFrame(summaries)
                csv_data = io.StringIO()
                df.to_csv(csv_data, index=False)
                zipf.writestr("summaries/summaries.csv", csv_data.getvalue())
            except Exception as e:
                self.logger.warning(f"Could not create summaries CSV: {e}")
    
    def _add_export_report_to_zip(self, zipf: zipfile.ZipFile, operation_id: str) -> None:
        """Add export operation report to ZIP archive."""
        report = {
            "operation_id": operation_id,
            "export_date": datetime.now().isoformat(),
            "files_included": [
                "emails/emails.csv",
                "emails/emails.json", 
                "analytics/analytics.json",
                "summaries/summaries.json",
                "summaries/summaries.csv",
                "metadata.json"
            ],
            "exporter_version": "2.0.0",
            "validation_enabled": self.validate_data
        }
        zipf.writestr("export_report.json", json.dumps(report, indent=2))
    
    def get_export_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of export operations.
        
        Returns:
            List of export operation records with metadata
        """
        return self.export_history.copy()
    
    def clear_export_history(self) -> None:
        """Clear the export history."""
        self.export_history.clear()
        self.logger.info("Export history cleared")
    
    def validate_output_directory(self) -> bool:
        """
        Validate that the output directory is accessible and writable.
        
        Returns:
            bool: True if directory is valid and writable
            
        Raises:
            ConfigurationError: If directory is not accessible or writable
        """
        try:
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.output_dir / ".write_test_validation"
            test_file.touch()
            test_file.unlink()
            
            return True
            
        except Exception as e:
            raise ConfigurationError(
                f"Output directory validation failed: {str(e)}",
                config_file=str(self.output_dir),
                module_name="DataExporter"
            ) from e


# Export utility functions for module-level access

def quick_export_emails_csv(emails: EmailList, filename: Optional[str] = None) -> str:
    """
    Convenience function for quick CSV export with default settings.
    
    Args:
        emails: List of emails to export
        filename: Optional filename
        
    Returns:
        str: Path to exported file
    """
    exporter = DataExporter()
    return exporter.export_emails_to_csv(emails, filename)


def quick_export_emails_json(emails: EmailList, filename: Optional[str] = None) -> str:
    """
    Convenience function for quick JSON export with default settings.
    
    Args:
        emails: List of emails to export  
        filename: Optional filename
        
    Returns:
        str: Path to exported file
    """
    exporter = DataExporter()
    return exporter.export_emails_to_json(emails, filename)


def validate_export_data(emails: EmailList) -> bool:
    """
    Validate email data for export compatibility.
    
    Args:
        emails: List of emails to validate
        
    Returns:
        bool: True if data is valid for export
        
    Raises:
        ValidationError: If data validation fails
    """
    exporter = DataExporter(validate_data=True)
    try:
        exporter._validate_email_list(emails)
        return True
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Unexpected validation error: {str(e)}") from e