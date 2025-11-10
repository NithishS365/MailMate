"""
Asynchronous Batch Processing for MailMate

This module provides scalable asynchronous batch processing capabilities using
Celery for distributed task execution. Features include:

- Email processing workflows (classification, summarization, etc.)
- Scalable task distribution across workers
- Task monitoring and progress tracking
- Retry mechanisms and error handling
- Result aggregation and storage
- Dynamic task scheduling and prioritization

Designed for high-throughput email processing with fault tolerance.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Union, Tuple, Callable,
    AsyncIterator, Generator, Type
)
import logging
import json
import pickle
import uuid
from pathlib import Path
# Optional imports with fallbacks
try:
    import redis
    HAS_REDIS = True
except ImportError:
    redis = None
    HAS_REDIS = False

try:
    from celery import Celery, Task, group, chord, chain
    HAS_CELERY = True
except ImportError:
    Celery = None
    Task = None
    group = None
    chord = None 
    chain = None
    HAS_CELERY = False
try:
    from celery.result import AsyncResult, GroupResult
    from celery.signals import task_prerun, task_postrun, task_failure
    from kombu import Queue
    HAS_CELERY_EXTRAS = True
except ImportError:
    AsyncResult = None
    GroupResult = None
    task_prerun = None
    task_postrun = None
    task_failure = None
    Queue = None
    HAS_CELERY_EXTRAS = False

from ..exceptions import (
    ValidationError, APIError, ConfigurationError, MailMateError,
    validate_string, validate_dict, validate_list
)
from ..logging_config import get_logger

logger = get_logger("mailmate.interfaces.batch_processing")


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingMode(Enum):
    """Batch processing modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    MAP_REDUCE = "map_reduce"


@dataclass
class BatchConfiguration:
    """
    Configuration for batch processing operations.
    
    Defines how batches should be processed, including concurrency,
    error handling, and result aggregation settings.
    """
    batch_id: str
    processing_mode: ProcessingMode = ProcessingMode.PARALLEL
    batch_size: int = 100
    max_workers: int = 4
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 3600  # seconds
    priority: TaskPriority = TaskPriority.NORMAL
    error_threshold: float = 0.1  # 10% error rate threshold
    result_storage: str = "redis"
    enable_progress_tracking: bool = True
    checkpoint_interval: int = 100  # items
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate batch configuration."""
        try:
            validate_string(self.batch_id, "batch_id", min_length=1, max_length=100)
            
            if not isinstance(self.processing_mode, ProcessingMode):
                raise ValidationError("processing_mode must be a ProcessingMode enum value")
            
            if not isinstance(self.priority, TaskPriority):
                raise ValidationError("priority must be a TaskPriority enum value")
            
            # Validate numeric constraints
            constraints = {
                'batch_size': (1, 10000),
                'max_workers': (1, 100),
                'max_retries': (0, 10),
                'retry_delay': (1, 3600),
                'timeout': (60, 86400),
                'checkpoint_interval': (1, 10000)
            }
            
            for field, (min_val, max_val) in constraints.items():
                value = getattr(self, field)
                if not isinstance(value, int) or value < min_val or value > max_val:
                    raise ValidationError(f"{field} must be an integer between {min_val} and {max_val}")
            
            if not 0 <= self.error_threshold <= 1:
                raise ValidationError("error_threshold must be between 0 and 1")
            
            if self.result_storage not in ["redis", "database", "file"]:
                raise ValidationError("result_storage must be 'redis', 'database', or 'file'")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Batch configuration validation failed: {str(e)}") from e


@dataclass 
class ProcessingMetrics:
    """
    Metrics for batch processing operations.
    
    Tracks performance, success rates, and processing statistics.
    """
    batch_id: str
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    retried_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time_seconds: float = 0.0
    throughput_items_per_second: float = 0.0
    error_rate: float = 0.0
    
    def update_progress(self, processed: int = 1, succeeded: bool = True) -> None:
        """Update processing progress."""
        self.processed_items += processed
        if succeeded:
            self.successful_items += processed
        else:
            self.failed_items += processed
        
        # Update metrics
        self._calculate_metrics()
    
    def mark_retry(self, count: int = 1) -> None:
        """Mark items as retried."""
        self.retried_items += count
    
    def start_processing(self) -> None:
        """Mark processing start time."""
        self.start_time = datetime.now()
    
    def end_processing(self) -> None:
        """Mark processing end time and calculate final metrics."""
        self.end_time = datetime.now()
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate derived metrics."""
        # Error rate
        if self.processed_items > 0:
            self.error_rate = self.failed_items / self.processed_items
        
        # Processing time and throughput
        if self.start_time:
            if self.end_time:
                duration = self.end_time - self.start_time
            else:
                duration = datetime.now() - self.start_time
            
            self.processing_time_seconds = duration.total_seconds()
            
            if self.processing_time_seconds > 0:
                self.throughput_items_per_second = self.processed_items / self.processing_time_seconds
    
    def get_progress_percent(self) -> float:
        """Get processing progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "batch_id": self.batch_id,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "retried_items": self.retried_items,
            "progress_percent": self.get_progress_percent(),
            "error_rate": self.error_rate,
            "throughput_items_per_second": self.throughput_items_per_second,
            "processing_time_seconds": self.processing_time_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class TaskQueue:
    """
    Task queue management with Celery backend.
    
    Provides high-level interface for task scheduling, monitoring,
    and result management.
    """
    
    def __init__(
        self, 
        broker_url: str = "redis://localhost:6379/0",
        result_backend: str = "redis://localhost:6379/0",
        app_name: str = "mailmate_tasks"
    ) -> None:
        """
        Initialize task queue.
        
        Args:
            broker_url: Message broker URL
            result_backend: Result storage backend URL
            app_name: Celery application name
        """
        if not HAS_CELERY or not HAS_REDIS:
            raise ConfigurationError(
                "Celery and Redis are required for TaskQueue. "
                "Install with: pip install celery redis"
            )
        
        try:
            # Validate inputs
            validate_string(broker_url, "broker_url", min_length=1)
            validate_string(result_backend, "result_backend", min_length=1)
            validate_string(app_name, "app_name", min_length=1, max_length=100)
            
            # Initialize Celery application
            self.app = Celery(app_name)
            self.app.conf.update(
                broker_url=broker_url,
                result_backend=result_backend,
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                # Task routing
                task_routes={
                    'mailmate.tasks.email.*': {'queue': 'email_processing'},
                    'mailmate.tasks.ml.*': {'queue': 'ml_inference'},
                    'mailmate.tasks.batch.*': {'queue': 'batch_processing'}
                },
                # Result expiration
                result_expires=3600,  # 1 hour
                # Worker configuration
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_disable_rate_limits=False
            )
            
            # Define queues
            self.app.conf.task_routes = {
                Queue('email_processing', routing_key='email'),
                Queue('ml_inference', routing_key='ml'),
                Queue('batch_processing', routing_key='batch'),
                Queue('high_priority', routing_key='priority')
            }
            
            self.logger = get_logger("mailmate.task_queue")
            self._redis_client = redis.Redis.from_url(result_backend)
            
            self.logger.info(f"TaskQueue initialized: {app_name}")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ConfigurationError(f"Failed to initialize TaskQueue: {str(e)}") from e
    
    def submit_task(
        self, 
        task_name: str,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        eta: Optional[datetime] = None,
        retry_policy: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_name: Name of the task to execute
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            priority: Task priority
            eta: Estimated time of arrival (scheduled execution)
            retry_policy: Custom retry policy
            
        Returns:
            str: Task ID
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            validate_string(task_name, "task_name", min_length=1)
            
            if kwargs is None:
                kwargs = {}
            
            # Determine queue based on priority
            queue_map = {
                TaskPriority.LOW: 'batch_processing',
                TaskPriority.NORMAL: 'email_processing',
                TaskPriority.HIGH: 'ml_inference',
                TaskPriority.CRITICAL: 'high_priority'
            }
            
            queue_name = queue_map.get(priority, 'email_processing')
            
            # Submit task
            result = self.app.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                queue=queue_name,
                eta=eta,
                retry_policy=retry_policy or {
                    'max_retries': 3,
                    'interval_start': 0,
                    'interval_step': 0.2,
                    'interval_max': 0.2
                }
            )
            
            task_id = result.id
            self.logger.debug(f"Submitted task {task_name} with ID {task_id}")
            
            return task_id
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise APIError(f"Failed to submit task {task_name}: {str(e)}") from e
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dict[str, Any]: Task status information
        """
        try:
            validate_string(task_id, "task_id", min_length=1)
            
            result = AsyncResult(task_id, app=self.app)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.successful() else None,
                "error": str(result.result) if result.failed() else None,
                "traceback": result.traceback if result.failed() else None,
                "date_done": result.date_done.isoformat() if result.date_done else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "UNKNOWN",
                "error": str(e)
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            bool: True if task was cancelled
        """
        try:
            validate_string(task_id, "task_id", min_length=1)
            
            result = AsyncResult(task_id, app=self.app)
            result.revoke(terminate=True)
            
            self.logger.info(f"Cancelled task {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            # Use Celery's inspect API
            inspect = self.app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active()
            
            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled()
            
            # Get worker stats
            stats = inspect.stats()
            
            return {
                "active_tasks": active_tasks,
                "scheduled_tasks": scheduled_tasks,
                "worker_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get queue stats: {e}")
            return {"error": str(e)}


class BatchProcessor:
    """
    High-level batch processing orchestrator.
    
    Manages complex batch processing workflows with error handling,
    progress tracking, and result aggregation.
    """
    
    def __init__(self, task_queue: TaskQueue) -> None:
        """
        Initialize batch processor.
        
        Args:
            task_queue: Task queue instance
        """
        if not isinstance(task_queue, TaskQueue):
            raise ValidationError("task_queue must be a TaskQueue instance")
        
        self.task_queue = task_queue
        self.logger = get_logger("mailmate.batch_processor")
        self._active_batches: Dict[str, ProcessingMetrics] = {}
        self._batch_results: Dict[str, Dict[str, Any]] = {}
    
    async def process_batch(
        self,
        items: List[Any],
        task_name: str,
        config: BatchConfiguration,
        item_preprocessor: Optional[Callable] = None,
        result_aggregator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of items asynchronously.
        
        Args:
            items: List of items to process
            task_name: Name of the Celery task to execute
            config: Batch processing configuration
            item_preprocessor: Optional preprocessor function
            result_aggregator: Optional result aggregation function
            
        Returns:
            Dict[str, Any]: Batch processing results
            
        Raises:
            ValidationError: If parameters are invalid
            APIError: If batch processing fails
        """
        try:
            # Validate inputs
            validate_list(items, "items")
            validate_string(task_name, "task_name", min_length=1)
            
            if not isinstance(config, BatchConfiguration):
                raise ValidationError("config must be a BatchConfiguration instance")
            
            if not items:
                return {"batch_id": config.batch_id, "status": "completed", "results": []}
            
            # Initialize metrics
            metrics = ProcessingMetrics(
                batch_id=config.batch_id,
                total_items=len(items)
            )
            metrics.start_processing()
            
            self._active_batches[config.batch_id] = metrics
            
            # Preprocess items if preprocessor provided
            if item_preprocessor:
                items = [item_preprocessor(item) for item in items]
            
            # Choose processing strategy
            if config.processing_mode == ProcessingMode.SEQUENTIAL:
                results = await self._process_sequential(items, task_name, config, metrics)
            elif config.processing_mode == ProcessingMode.PARALLEL:
                results = await self._process_parallel(items, task_name, config, metrics)
            elif config.processing_mode == ProcessingMode.PIPELINE:
                results = await self._process_pipeline(items, task_name, config, metrics)
            elif config.processing_mode == ProcessingMode.MAP_REDUCE:
                results = await self._process_map_reduce(items, task_name, config, metrics)
            else:
                raise ValidationError(f"Unsupported processing mode: {config.processing_mode}")
            
            # Aggregate results if aggregator provided
            if result_aggregator:
                aggregated_results = result_aggregator(results)
            else:
                aggregated_results = results
            
            metrics.end_processing()
            
            # Store final results
            batch_result = {
                "batch_id": config.batch_id,
                "status": "completed" if metrics.error_rate <= config.error_threshold else "completed_with_errors",
                "metrics": metrics.to_dict(),
                "results": aggregated_results,
                "config": {
                    "processing_mode": config.processing_mode.value,
                    "batch_size": config.batch_size,
                    "max_workers": config.max_workers
                }
            }
            
            self._batch_results[config.batch_id] = batch_result
            
            self.logger.info(
                f"Batch {config.batch_id} completed: "
                f"{metrics.successful_items}/{metrics.total_items} succeeded, "
                f"{metrics.error_rate:.2%} error rate"
            )
            
            return batch_result
            
        except Exception as e:
            if config.batch_id in self._active_batches:
                self._active_batches[config.batch_id].end_processing()
            
            if isinstance(e, (ValidationError, APIError)):
                raise
            raise APIError(f"Batch processing failed for {config.batch_id}: {str(e)}") from e
    
    async def _process_sequential(
        self,
        items: List[Any],
        task_name: str,
        config: BatchConfiguration,
        metrics: ProcessingMetrics
    ) -> List[Any]:
        """Process items sequentially."""
        results = []
        
        for i, item in enumerate(items):
            try:
                task_id = self.task_queue.submit_task(
                    task_name,
                    args=(item,),
                    priority=config.priority
                )
                
                # Wait for task completion
                result = await self._wait_for_task(task_id, config.timeout)
                results.append(result)
                
                metrics.update_progress(succeeded=True)
                
                # Checkpoint progress
                if config.enable_progress_tracking and (i + 1) % config.checkpoint_interval == 0:
                    self.logger.info(f"Batch {config.batch_id}: {i + 1}/{len(items)} items processed")
                
            except Exception as e:
                self.logger.error(f"Item {i} failed in batch {config.batch_id}: {e}")
                results.append({"error": str(e)})
                metrics.update_progress(succeeded=False)
                
                # Check error threshold
                if metrics.error_rate > config.error_threshold:
                    self.logger.error(f"Error threshold exceeded for batch {config.batch_id}")
                    break
        
        return results
    
    async def _process_parallel(
        self,
        items: List[Any],
        task_name: str,
        config: BatchConfiguration,
        metrics: ProcessingMetrics
    ) -> List[Any]:
        """Process items in parallel batches."""
        results = []
        
        # Split items into batches
        batches = [items[i:i + config.batch_size] for i in range(0, len(items), config.batch_size)]
        
        for batch_items in batches:
            # Submit all tasks in this batch
            task_ids = []
            for item in batch_items:
                task_id = self.task_queue.submit_task(
                    task_name,
                    args=(item,),
                    priority=config.priority
                )
                task_ids.append(task_id)
            
            # Wait for all tasks in this batch to complete
            batch_results = await asyncio.gather(
                *[self._wait_for_task(task_id, config.timeout) for task_id in task_ids],
                return_exceptions=True
            )
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"error": str(result)})
                    metrics.update_progress(succeeded=False)
                else:
                    results.append(result)
                    metrics.update_progress(succeeded=True)
            
            # Check error threshold
            if metrics.error_rate > config.error_threshold:
                self.logger.error(f"Error threshold exceeded for batch {config.batch_id}")
                break
            
            # Progress update
            if config.enable_progress_tracking:
                self.logger.info(
                    f"Batch {config.batch_id}: {metrics.processed_items}/{metrics.total_items} items processed"
                )
        
        return results
    
    async def _process_pipeline(
        self,
        items: List[Any],
        task_name: str,
        config: BatchConfiguration,
        metrics: ProcessingMetrics
    ) -> List[Any]:
        """Process items using pipeline pattern."""
        # For pipeline processing, we'd typically have multiple stages
        # This is a simplified implementation
        return await self._process_parallel(items, task_name, config, metrics)
    
    async def _process_map_reduce(
        self,
        items: List[Any],
        task_name: str,
        config: BatchConfiguration,
        metrics: ProcessingMetrics
    ) -> List[Any]:
        """Process items using map-reduce pattern."""
        # Map phase - process all items
        map_results = await self._process_parallel(items, task_name, config, metrics)
        
        # Reduce phase - aggregate results (simplified)
        # In a real implementation, this would involve a separate reduce task
        successful_results = [r for r in map_results if not isinstance(r, dict) or "error" not in r]
        
        return {
            "map_results": map_results,
            "reduced_results": successful_results,
            "total_successful": len(successful_results),
            "total_failed": len(map_results) - len(successful_results)
        }
    
    async def _wait_for_task(self, task_id: str, timeout: int) -> Any:
        """Wait for a task to complete with timeout."""
        start_time = datetime.now()
        
        while True:
            status = self.task_queue.get_task_status(task_id)
            
            if status["status"] in ["SUCCESS", "FAILURE", "REVOKED"]:
                if status["status"] == "SUCCESS":
                    return status["result"]
                else:
                    raise Exception(status.get("error", "Task failed"))
            
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > timeout:
                self.task_queue.cancel_task(task_id)
                raise Exception(f"Task {task_id} timed out after {timeout} seconds")
            
            # Wait before polling again
            await asyncio.sleep(1)
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch."""
        if batch_id in self._batch_results:
            return self._batch_results[batch_id]
        
        if batch_id in self._active_batches:
            metrics = self._active_batches[batch_id]
            return {
                "batch_id": batch_id,
                "status": "in_progress",
                "metrics": metrics.to_dict()
            }
        
        return None
    
    def list_active_batches(self) -> List[str]:
        """List currently active batch IDs."""
        return list(self._active_batches.keys())
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        total_batches = len(self._batch_results) + len(self._active_batches)
        completed_batches = len(self._batch_results)
        active_batches = len(self._active_batches)
        
        return {
            "total_batches": total_batches,
            "completed_batches": completed_batches,
            "active_batches": active_batches,
            "completion_rate": (completed_batches / total_batches * 100) if total_batches > 0 else 0
        }


class AsyncEmailProcessor:
    """
    Specialized async processor for email-related tasks.
    
    Provides high-level interface for common email processing workflows
    like classification, summarization, and content analysis.
    """
    
    def __init__(self, batch_processor: BatchProcessor) -> None:
        """Initialize async email processor."""
        if not isinstance(batch_processor, BatchProcessor):
            raise ValidationError("batch_processor must be a BatchProcessor instance")
        
        self.batch_processor = batch_processor
        self.logger = get_logger("mailmate.async_email_processor")
    
    async def classify_emails_batch(
        self,
        emails: List[Dict[str, Any]],
        model_id: str,
        batch_config: Optional[BatchConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Classify a batch of emails asynchronously.
        
        Args:
            emails: List of email data dictionaries
            model_id: ID of the classification model to use
            batch_config: Optional batch configuration
            
        Returns:
            Dict[str, Any]: Classification results
        """
        if not batch_config:
            batch_config = BatchConfiguration(
                batch_id=f"classify_{uuid.uuid4().hex[:8]}",
                processing_mode=ProcessingMode.PARALLEL,
                batch_size=50
            )
        
        def preprocess_email(email):
            return {"email_data": email, "model_id": model_id}
        
        return await self.batch_processor.process_batch(
            items=emails,
            task_name="mailmate.tasks.ml.classify_email",
            config=batch_config,
            item_preprocessor=preprocess_email
        )
    
    async def summarize_emails_batch(
        self,
        emails: List[Dict[str, Any]],
        model_id: str,
        batch_config: Optional[BatchConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Summarize a batch of emails asynchronously.
        
        Args:
            emails: List of email data dictionaries
            model_id: ID of the summarization model to use
            batch_config: Optional batch configuration
            
        Returns:
            Dict[str, Any]: Summarization results
        """
        if not batch_config:
            batch_config = BatchConfiguration(
                batch_id=f"summarize_{uuid.uuid4().hex[:8]}",
                processing_mode=ProcessingMode.PARALLEL,
                batch_size=20  # Smaller batches for summarization
            )
        
        def preprocess_email(email):
            return {"email_data": email, "model_id": model_id}
        
        return await self.batch_processor.process_batch(
            items=emails,
            task_name="mailmate.tasks.ml.summarize_email",
            config=batch_config,
            item_preprocessor=preprocess_email
        )
    
    async def process_email_workflow(
        self,
        emails: List[Dict[str, Any]],
        workflow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process emails through a complete workflow (classify -> summarize -> export).
        
        Args:
            emails: List of email data dictionaries
            workflow_config: Workflow configuration
            
        Returns:
            Dict[str, Any]: Complete workflow results
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        try:
            results = {"workflow_id": workflow_id, "stages": {}}
            
            # Stage 1: Classification
            if workflow_config.get("classify", False):
                classify_config = BatchConfiguration(
                    batch_id=f"{workflow_id}_classify",
                    processing_mode=ProcessingMode.PARALLEL
                )
                
                classify_results = await self.classify_emails_batch(
                    emails,
                    workflow_config["classification_model"],
                    classify_config
                )
                results["stages"]["classification"] = classify_results
            
            # Stage 2: Summarization
            if workflow_config.get("summarize", False):
                summarize_config = BatchConfiguration(
                    batch_id=f"{workflow_id}_summarize",
                    processing_mode=ProcessingMode.PARALLEL
                )
                
                summarize_results = await self.summarize_emails_batch(
                    emails,
                    workflow_config["summarization_model"],
                    summarize_config
                )
                results["stages"]["summarization"] = summarize_results
            
            # Stage 3: Additional processing stages can be added here
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            raise APIError(f"Email workflow processing failed: {str(e)}") from e


# Global instances
default_task_queue = TaskQueue()
default_batch_processor = BatchProcessor(default_task_queue)
async_email_processor = AsyncEmailProcessor(default_batch_processor)