"""
Machine Learning Model Interfaces for MailMate

This module provides extensible base classes and interfaces for integrating
machine learning models into the MailMate system. Supports:

- Classification models (email categorization, priority detection)
- Summarization models (email content summarization)
- Pluggable model architecture with runtime loading
- Model performance monitoring and metrics
- Configuration management and validation

All models follow a consistent interface for easy swapping and testing.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Union, Tuple, Callable,
    Generic, TypeVar, Protocol, runtime_checkable
)
import logging
import json
import pickle
from contextlib import contextmanager

from ..exceptions import (
    ValidationError, ClassificationError, SummarizationError,
    ConfigurationError, MailMateError, validate_string, validate_dict
)
from ..logging_config import get_logger

logger = get_logger("mailmate.interfaces.ml_models")

# Type definitions
ModelInput = TypeVar('ModelInput')
ModelOutput = TypeVar('ModelOutput')
EmailData = Dict[str, Any]  # Will match existing email data structure


class ModelType(Enum):
    """Supported model types."""
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PRIORITY_DETECTION = "priority_detection"
    LANGUAGE_DETECTION = "language_detection"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """Model status indicators."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    TRAINING = "training"
    UPDATING = "updating"


@dataclass
class ModelConfiguration:
    """
    Configuration for ML models with validation.
    
    Attributes:
        model_id: Unique identifier for the model
        model_type: Type of ML model
        model_path: Path to model files
        config_params: Model-specific configuration parameters
        version: Model version
        description: Human-readable description
        required_features: List of required input features
        optional_features: List of optional input features
        performance_thresholds: Minimum performance requirements
        resource_limits: Resource usage limits
    """
    model_id: str
    model_type: ModelType
    model_path: Optional[str] = None
    config_params: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    description: str = ""
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate all configuration fields."""
        try:
            # Validate required string fields
            validate_string(self.model_id, "model_id", min_length=1, max_length=100)
            validate_string(self.version, "version", min_length=1, max_length=50)
            
            # Validate model type
            if not isinstance(self.model_type, ModelType):
                raise ValidationError(f"model_type must be a ModelType enum value")
            
            # Validate dictionaries
            validate_dict(self.config_params, "config_params")
            validate_dict(self.performance_thresholds, "performance_thresholds")
            validate_dict(self.resource_limits, "resource_limits")
            
            # Validate model path if provided
            if self.model_path:
                validate_string(self.model_path, "model_path", min_length=1, max_length=500)
                if not Path(self.model_path).exists():
                    logger.warning(f"Model path does not exist: {self.model_path}")
            
            # Validate feature lists
            if not isinstance(self.required_features, list):
                raise ValidationError("required_features must be a list")
            if not isinstance(self.optional_features, list):
                raise ValidationError("optional_features must be a list")
            
            for i, feature in enumerate(self.required_features):
                validate_string(feature, f"required_features[{i}]", min_length=1, max_length=100)
            
            for i, feature in enumerate(self.optional_features):
                validate_string(feature, f"optional_features[{i}]", min_length=1, max_length=100)
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Model configuration validation failed: {str(e)}") from e


@dataclass
class ModelMetrics:
    """
    Performance metrics for ML models.
    
    Tracks accuracy, performance, and usage statistics.
    """
    model_id: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate prediction success rate."""
        if self.total_predictions == 0:
            return 0.0
        return (self.successful_predictions / self.total_predictions) * 100


@runtime_checkable
class ModelInterface(Protocol, Generic[ModelInput, ModelOutput]):
    """Protocol defining the interface all ML models must implement."""
    
    def predict(self, input_data: ModelInput) -> ModelOutput:
        """Make a prediction on input data."""
        ...
    
    async def predict_async(self, input_data: ModelInput) -> ModelOutput:
        """Make an asynchronous prediction."""
        ...
    
    def predict_batch(self, input_batch: List[ModelInput]) -> List[ModelOutput]:
        """Make predictions on a batch of inputs."""
        ...
    
    async def predict_batch_async(self, input_batch: List[ModelInput]) -> List[ModelOutput]:
        """Make asynchronous batch predictions."""
        ...


class BaseMLModel(ABC, Generic[ModelInput, ModelOutput]):
    """
    Abstract base class for all ML models in MailMate.
    
    Provides common functionality and enforces consistent interface
    across all model implementations.
    """
    
    def __init__(self, config: ModelConfiguration) -> None:
        """
        Initialize the model with configuration.
        
        Args:
            config: Model configuration object
            
        Raises:
            ValidationError: If configuration is invalid
            ConfigurationError: If model setup fails
        """
        try:
            if not isinstance(config, ModelConfiguration):
                raise ValidationError("config must be a ModelConfiguration instance")
            
            self.config = config
            self.model_id = config.model_id
            self.model_type = config.model_type
            self.status = ModelStatus.LOADING
            self.metrics = ModelMetrics(model_id=config.model_id)
            self.logger = get_logger(f"mailmate.model.{self.model_id}")
            
            # Model state
            self._model = None
            self._is_loaded = False
            self._load_time = None
            
            # Initialize model-specific components
            self._initialize_model()
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(f"Failed to initialize model {config.model_id}: {str(e)}") from e
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize model-specific components. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Load the actual model. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _preprocess_input(self, input_data: ModelInput) -> Any:
        """Preprocess input data. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _postprocess_output(self, raw_output: Any) -> ModelOutput:
        """Postprocess model output. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict_internal(self, preprocessed_input: Any) -> Any:
        """Internal prediction method. Implemented by subclasses."""
        pass
    
    def load(self) -> None:
        """
        Load the model into memory.
        
        Raises:
            ConfigurationError: If model loading fails
        """
        if self._is_loaded:
            return
        
        try:
            self.status = ModelStatus.LOADING
            start_time = datetime.now()
            
            self._model = self._load_model()
            
            self._load_time = datetime.now() - start_time
            self._is_loaded = True
            self.status = ModelStatus.READY
            
            self.logger.info(f"Model {self.model_id} loaded successfully in {self._load_time.total_seconds():.2f}s")
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            self._is_loaded = False
            raise ConfigurationError(f"Failed to load model {self.model_id}: {str(e)}") from e
    
    def unload(self) -> None:
        """Unload the model from memory."""
        try:
            if self._model is not None:
                # Clean up model resources
                if hasattr(self._model, 'close'):
                    self._model.close()
                del self._model
                self._model = None
            
            self._is_loaded = False
            self.status = ModelStatus.LOADING
            
            self.logger.info(f"Model {self.model_id} unloaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Error unloading model {self.model_id}: {e}")
    
    @contextmanager
    def _prediction_context(self):
        """Context manager for prediction operations with metrics tracking."""
        start_time = datetime.now()
        success = False
        
        try:
            if not self._is_loaded:
                self.load()
            
            yield
            success = True
            
        except Exception as e:
            self.metrics.failed_predictions += 1
            raise
        
        finally:
            # Update metrics
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
            
            self.metrics.total_predictions += 1
            if success:
                self.metrics.successful_predictions += 1
            
            # Update average inference time
            if self.metrics.total_predictions > 1:
                self.metrics.inference_time_ms = (
                    (self.metrics.inference_time_ms * (self.metrics.total_predictions - 1) + inference_time) 
                    / self.metrics.total_predictions
                )
            else:
                self.metrics.inference_time_ms = inference_time
            
            self.metrics.last_updated = end_time
    
    def predict(self, input_data: ModelInput) -> ModelOutput:
        """
        Make a prediction on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            ModelOutput: Prediction result
            
        Raises:
            ValidationError: If input is invalid
            ClassificationError: If prediction fails
        """
        try:
            with self._prediction_context():
                # Preprocess input
                preprocessed = self._preprocess_input(input_data)
                
                # Make prediction
                raw_output = self._predict_internal(preprocessed)
                
                # Postprocess output
                result = self._postprocess_output(raw_output)
                
                return result
                
        except Exception as e:
            if isinstance(e, (ValidationError, ClassificationError)):
                raise
            raise ClassificationError(f"Prediction failed for model {self.model_id}: {str(e)}") from e
    
    async def predict_async(self, input_data: ModelInput) -> ModelOutput:
        """
        Make an asynchronous prediction.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            ModelOutput: Prediction result
        """
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, input_data)
    
    def predict_batch(self, input_batch: List[ModelInput]) -> List[ModelOutput]:
        """
        Make predictions on a batch of inputs.
        
        Args:
            input_batch: List of input data
            
        Returns:
            List[ModelOutput]: List of prediction results
        """
        if not input_batch:
            return []
        
        results = []
        for input_data in input_batch:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch prediction failed for item: {e}")
                # Could append None or a default error result
                results.append(None)
        
        return results
    
    async def predict_batch_async(self, input_batch: List[ModelInput]) -> List[ModelOutput]:
        """
        Make asynchronous batch predictions.
        
        Args:
            input_batch: List of input data
            
        Returns:
            List[ModelOutput]: List of prediction results
        """
        if not input_batch:
            return []
        
        # Create tasks for concurrent processing
        tasks = [self.predict_async(input_data) for input_data in input_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Async batch prediction failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "status": self.status.value,
            "is_loaded": self._is_loaded,
            "load_time": self._load_time.total_seconds() if self._load_time else None,
            "version": self.config.version,
            "description": self.config.description,
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "total_predictions": self.metrics.total_predictions,
                "success_rate": self.metrics.get_success_rate(),
                "avg_inference_time_ms": self.metrics.inference_time_ms,
                "last_updated": self.metrics.last_updated.isoformat()
            }
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update model configuration.
        
        Args:
            new_config: New configuration parameters
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate_dict(new_config, "new_config")
            
            # Update config params
            self.config.config_params.update(new_config)
            
            # Re-validate configuration
            self.config._validate_config()
            
            self.logger.info(f"Updated configuration for model {self.model_id}")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to update config for model {self.model_id}: {str(e)}") from e


class ClassificationModel(BaseMLModel[EmailData, Dict[str, Any]]):
    """
    Base class for email classification models.
    
    Handles email categorization, priority detection, and other
    classification tasks specific to email processing.
    """
    
    def __init__(self, config: ModelConfiguration) -> None:
        """Initialize classification model."""
        if config.model_type not in [ModelType.CLASSIFICATION, ModelType.PRIORITY_DETECTION]:
            raise ValidationError(f"Invalid model type for ClassificationModel: {config.model_type}")
        
        super().__init__(config)
    
    def _initialize_model(self) -> None:
        """Initialize classification-specific components."""
        self.classes = self.config.config_params.get('classes', [])
        self.confidence_threshold = self.config.config_params.get('confidence_threshold', 0.5)
        
        if not self.classes:
            self.logger.warning(f"No classes defined for classification model {self.model_id}")
    
    def _preprocess_input(self, input_data: EmailData) -> Dict[str, Any]:
        """Preprocess email data for classification."""
        try:
            # Validate input
            if not isinstance(input_data, dict):
                raise ValidationError("Input must be a dictionary containing email data")
            
            # Extract text features
            text_features = []
            for field in ['subject', 'body', 'from_address']:
                if field in input_data and input_data[field]:
                    text_features.append(str(input_data[field]))
            
            if not text_features:
                raise ValidationError("No text content found in email data")
            
            combined_text = ' '.join(text_features)
            
            return {
                'text': combined_text,
                'subject': input_data.get('subject', ''),
                'body': input_data.get('body', ''),
                'from_address': input_data.get('from_address', ''),
                'metadata': {k: v for k, v in input_data.items() 
                           if k not in ['subject', 'body', 'from_address']}
            }
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to preprocess email data: {str(e)}") from e
    
    def _postprocess_output(self, raw_output: Any) -> Dict[str, Any]:
        """Postprocess classification output."""
        try:
            # Ensure output is in expected format
            if isinstance(raw_output, dict):
                result = raw_output.copy()
            else:
                # Convert to standard format
                result = {'prediction': raw_output}
            
            # Add standard fields
            if 'confidence' not in result:
                result['confidence'] = 1.0 if 'probabilities' not in result else max(result.get('probabilities', {}).values())
            
            if 'model_id' not in result:
                result['model_id'] = self.model_id
            
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            raise ClassificationError(f"Failed to postprocess classification output: {str(e)}") from e


class SummarizationModel(BaseMLModel[EmailData, Dict[str, Any]]):
    """
    Base class for email summarization models.
    
    Handles email content summarization with various strategies
    and output formats.
    """
    
    def __init__(self, config: ModelConfiguration) -> None:
        """Initialize summarization model."""
        if config.model_type != ModelType.SUMMARIZATION:
            raise ValidationError(f"Invalid model type for SummarizationModel: {config.model_type}")
        
        super().__init__(config)
    
    def _initialize_model(self) -> None:
        """Initialize summarization-specific components."""
        self.max_length = self.config.config_params.get('max_length', 150)
        self.min_length = self.config.config_params.get('min_length', 30)
        self.strategy = self.config.config_params.get('strategy', 'extractive')
        
        if self.strategy not in ['extractive', 'abstractive', 'hybrid']:
            raise ConfigurationError(f"Invalid summarization strategy: {self.strategy}")
    
    def _preprocess_input(self, input_data: EmailData) -> Dict[str, Any]:
        """Preprocess email data for summarization."""
        try:
            # Validate input
            if not isinstance(input_data, dict):
                raise ValidationError("Input must be a dictionary containing email data")
            
            # Extract content for summarization
            content_parts = []
            
            if 'subject' in input_data and input_data['subject']:
                content_parts.append(f"Subject: {input_data['subject']}")
            
            if 'body' in input_data and input_data['body']:
                content_parts.append(input_data['body'])
            
            if not content_parts:
                raise ValidationError("No content found for summarization")
            
            full_content = '\n'.join(content_parts)
            
            return {
                'content': full_content,
                'subject': input_data.get('subject', ''),
                'body': input_data.get('body', ''),
                'length_constraints': {
                    'max_length': self.max_length,
                    'min_length': self.min_length
                },
                'strategy': self.strategy,
                'metadata': input_data
            }
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to preprocess email for summarization: {str(e)}") from e
    
    def _postprocess_output(self, raw_output: Any) -> Dict[str, Any]:
        """Postprocess summarization output."""
        try:
            # Ensure output is in expected format
            if isinstance(raw_output, str):
                summary = raw_output
                metadata = {}
            elif isinstance(raw_output, dict):
                summary = raw_output.get('summary', raw_output.get('text', ''))
                metadata = {k: v for k, v in raw_output.items() if k not in ['summary', 'text']}
            else:
                summary = str(raw_output)
                metadata = {}
            
            if not summary:
                raise SummarizationError("Generated summary is empty")
            
            result = {
                'summary': summary,
                'length': len(summary),
                'model_id': self.model_id,
                'strategy': self.strategy,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }
            
            return result
            
        except Exception as e:
            if isinstance(e, SummarizationError):
                raise
            raise SummarizationError(f"Failed to postprocess summarization output: {str(e)}") from e


class ModelRegistry:
    """
    Registry for managing ML models with thread-safe operations.
    
    Provides centralized model management, loading, and discovery.
    """
    
    def __init__(self) -> None:
        """Initialize the model registry."""
        self._models: Dict[str, BaseMLModel] = {}
        self._model_configs: Dict[str, ModelConfiguration] = {}
        self._lock = asyncio.Lock()
        self.logger = get_logger("mailmate.model_registry")
    
    async def register_model(self, model: BaseMLModel) -> None:
        """
        Register a model in the registry.
        
        Args:
            model: Model instance to register
            
        Raises:
            ValidationError: If model is invalid
        """
        async with self._lock:
            if not isinstance(model, BaseMLModel):
                raise ValidationError("model must be a BaseMLModel instance")
            
            model_id = model.model_id
            
            if model_id in self._models:
                self.logger.warning(f"Model {model_id} already registered, replacing")
            
            self._models[model_id] = model
            self._model_configs[model_id] = model.config
            
            self.logger.info(f"Registered model: {model_id} ({model.model_type.value})")
    
    async def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            bool: True if model was unregistered
        """
        async with self._lock:
            if model_id in self._models:
                # Unload model before removing
                model = self._models[model_id]
                model.unload()
                
                del self._models[model_id]
                del self._model_configs[model_id]
                
                self.logger.info(f"Unregistered model: {model_id}")
                return True
            
            return False
    
    def get_model(self, model_id: str) -> Optional[BaseMLModel]:
        """Get a model by ID."""
        return self._models.get(model_id)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[str]:
        """
        List registered models, optionally filtered by type.
        
        Args:
            model_type: Optional model type filter
            
        Returns:
            List[str]: List of model IDs
        """
        if model_type is None:
            return list(self._models.keys())
        
        return [
            model_id for model_id, model in self._models.items()
            if model.model_type == model_type
        ]
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        model = self.get_model(model_id)
        return model.get_info() if model else None
    
    async def predict_with_model(
        self, 
        model_id: str, 
        input_data: Any, 
        async_mode: bool = False
    ) -> Any:
        """
        Make a prediction using a registered model.
        
        Args:
            model_id: Model identifier
            input_data: Input data for prediction
            async_mode: Whether to use async prediction
            
        Returns:
            Prediction result
            
        Raises:
            ValidationError: If model not found
            ClassificationError: If prediction fails
        """
        model = self.get_model(model_id)
        if not model:
            raise ValidationError(f"Model not found: {model_id}")
        
        if async_mode:
            return await model.predict_async(input_data)
        else:
            return model.predict(input_data)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        model_types = {}
        total_predictions = 0
        
        for model in self._models.values():
            model_type = model.model_type.value
            model_types[model_type] = model_types.get(model_type, 0) + 1
            total_predictions += model.metrics.total_predictions
        
        return {
            "total_models": len(self._models),
            "model_types": model_types,
            "total_predictions": total_predictions,
            "registered_models": list(self._models.keys())
        }


# Global model registry instance
model_registry = ModelRegistry()