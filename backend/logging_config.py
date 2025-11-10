"""
MailMate Logging Configuration Module

Provides centralized logging setup for all MailMate modules with configurable
levels, file rotation, and structured formatting.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class MailMateLogger:
    """Centralized logging configuration for MailMate application."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 log_format: Optional[str] = None):
        """
        Initialize the logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            max_file_size: Maximum size of each log file in bytes
            backup_count: Number of backup files to keep
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            log_format: Custom log format string
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # Default log format
        if log_format is None:
            self.log_format = (
                "%(asctime)s | %(name)s | %(levelname)s | "
                "%(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
            )
        else:
            self.log_format = log_format
            
        # Ensure log directory exists
        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup the root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            self.log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file:
            log_file = self.log_dir / "mailmate.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def create_module_logger(self, 
                           module_name: str, 
                           separate_file: bool = False,
                           file_level: Optional[str] = None) -> logging.Logger:
        """
        Create a specialized logger for a specific module.
        
        Args:
            module_name: Name of the module
            separate_file: Whether to create a separate log file
            file_level: Specific log level for file output
            
        Returns:
            Configured module logger
        """
        logger = self.get_logger(module_name)
        
        if separate_file and self.enable_file:
            # Create separate file handler for this module
            module_log_file = self.log_dir / f"{module_name}.log"
            module_file_handler = logging.handlers.RotatingFileHandler(
                module_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            
            # Set file-specific level if provided
            if file_level:
                module_file_handler.setLevel(getattr(logging, file_level.upper()))
            else:
                module_file_handler.setLevel(self.log_level)
            
            # Create module-specific formatter
            module_formatter = logging.Formatter(
                f"%(asctime)s | {module_name.upper()} | %(levelname)s | "
                "%(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            module_file_handler.setFormatter(module_formatter)
            logger.addHandler(module_file_handler)
        
        return logger
    
    def log_performance(self, logger: logging.Logger, operation: str, duration: float):
        """
        Log performance metrics for operations.
        
        Args:
            logger: Logger instance to use
            operation: Name of the operation
            duration: Duration in seconds
        """
        performance_data = {
            "operation": operation,
            "duration_seconds": round(duration, 4),
            "timestamp": datetime.now().isoformat(),
            "type": "performance"
        }
        
        logger.info(f"PERFORMANCE: {operation} completed in {duration:.4f}s", 
                   extra={"performance_data": performance_data})
    
    def log_error_with_context(self, 
                             logger: logging.Logger, 
                             error: Exception, 
                             context: Dict[str, Any]):
        """
        Log errors with additional context information.
        
        Args:
            logger: Logger instance to use
            error: Exception that occurred
            context: Additional context information
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "type": "error"
        }
        
        logger.error(f"ERROR: {type(error).__name__}: {str(error)}", 
                    exc_info=True, 
                    extra={"error_data": error_data})
    
    def set_level(self, level: str):
        """
        Change the logging level for all loggers.
        
        Args:
            level: New logging level
        """
        new_level = getattr(logging, level.upper())
        self.log_level = new_level
        
        # Update root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(new_level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(new_level)
        
        # Update module loggers
        for logger in self._loggers.values():
            logger.setLevel(new_level)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about log files and logging activity.
        
        Returns:
            Dictionary containing log statistics
        """
        stats = {
            "log_level": logging.getLevelName(self.log_level),
            "log_directory": str(self.log_dir),
            "console_enabled": self.enable_console,
            "file_enabled": self.enable_file,
            "active_loggers": list(self._loggers.keys()),
            "log_files": []
        }
        
        if self.enable_file and self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log"):
                file_stats = log_file.stat()
                stats["log_files"].append({
                    "name": log_file.name,
                    "size_bytes": file_stats.st_size,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
        
        return stats


# Global logger instance
_logger_instance: Optional[MailMateLogger] = None


def setup_logging(config: Optional[Dict[str, Any]] = None) -> MailMateLogger:
    """
    Setup global logging configuration for MailMate.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MailMateLogger instance
    """
    global _logger_instance
    
    if config is None:
        config = {}
    
    _logger_instance = MailMateLogger(**config)
    return _logger_instance


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    if _logger_instance is None:
        setup_logging()
    
    return _logger_instance.get_logger(name)


def get_logger_instance() -> Optional[MailMateLogger]:
    """Get the global logger instance."""
    return _logger_instance


# Convenience functions for common logging patterns
def log_startup(logger: logging.Logger, component: str, version: str = "1.0.0"):
    """Log application startup."""
    logger.info(f"Starting {component} v{version}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")


def log_shutdown(logger: logging.Logger, component: str):
    """Log application shutdown."""
    logger.info(f"Shutting down {component}")


def log_config_loaded(logger: logging.Logger, config_file: str, config_data: Dict[str, Any]):
    """Log configuration loading."""
    logger.info(f"Configuration loaded from: {config_file}")
    logger.debug(f"Configuration data: {json.dumps(config_data, indent=2, default=str)}")


def log_api_request(logger: logging.Logger, 
                   method: str, 
                   endpoint: str, 
                   status_code: int,
                   duration: float):
    """Log API request details."""
    logger.info(f"API {method} {endpoint} - {status_code} ({duration:.3f}s)")


def log_email_processed(logger: logging.Logger, 
                       email_id: str, 
                       operation: str, 
                       success: bool,
                       details: Optional[str] = None):
    """Log email processing operations."""
    status = "SUCCESS" if success else "FAILED"
    message = f"Email {operation}: {email_id} - {status}"
    if details:
        message += f" | {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging with custom configuration
    config = {
        "log_level": "DEBUG",
        "log_dir": "logs",
        "enable_console": True,
        "enable_file": True
    }
    
    logger_manager = setup_logging(config)
    
    # Create different module loggers
    main_logger = get_logger("mailmate.main")
    api_logger = logger_manager.create_module_logger("api", separate_file=True)
    email_logger = logger_manager.create_module_logger("email_processor", separate_file=True)
    
    # Test logging
    log_startup(main_logger, "MailMate Dashboard", "1.0.0")
    
    main_logger.debug("This is a debug message")
    main_logger.info("This is an info message")
    main_logger.warning("This is a warning message")
    
    # Test error logging with context
    try:
        raise ValueError("Test error for demonstration")
    except Exception as e:
        logger_manager.log_error_with_context(
            main_logger, 
            e, 
            {"operation": "test", "user_id": "demo"}
        )
    
    # Test performance logging
    import time
    start_time = time.time()
    time.sleep(0.1)  # Simulate work
    duration = time.time() - start_time
    logger_manager.log_performance(main_logger, "test_operation", duration)
    
    # Log email processing
    log_email_processed(email_logger, "email_123", "classification", True, "Category: Work")
    
    # Log API request
    log_api_request(api_logger, "GET", "/api/emails", 200, 0.045)
    
    # Print log statistics
    stats = logger_manager.get_log_stats()
    main_logger.info(f"Logging statistics: {json.dumps(stats, indent=2)}")
    
    log_shutdown(main_logger, "MailMate Dashboard")