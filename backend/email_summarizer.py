"""
EmailSummarizer module for MailMate - AI-driven email summarization system.

This module provides the EmailSummarizer class that supports:
- Text summarization using HuggingFace transformer models (BART, Pegasus, T5)
- Automatic model loading and caching
- Configurable summary length and quality parameters
- Batch summarization for multiple emails
- Text preprocessing and post-processing
- Performance optimization and error handling
"""

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# HuggingFace imports (with graceful fallback)
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        pipeline,
        BartTokenizer, BartForConditionalGeneration,
        PegasusTokenizer, PegasusForConditionalGeneration,
        T5Tokenizer, T5ForConditionalGeneration
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("HuggingFace transformers not available. Install with: pip install transformers torch")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Result of email summarization."""
    original_text: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: float
    model_used: str
    confidence_score: Optional[float] = None


class EmailSummarizerError(Exception):
    """Base exception for EmailSummarizer errors."""
    pass


class ModelNotAvailableError(EmailSummarizerError):
    """Exception raised when required model is not available."""
    pass


class TextTooShortError(EmailSummarizerError):
    """Exception raised when text is too short to summarize."""
    pass


class EmailSummarizer:
    """
    EmailSummarizer for generating concise summaries from email content.
    
    Supports multiple transformer models:
    - BART: Bidirectional Auto-Regressive Transformers
    - Pegasus: Pre-trained for abstractive summarization
    - T5: Text-to-Text Transfer Transformer
    
    Features:
    - Automatic model loading and caching
    - Configurable summary parameters
    - Text preprocessing and cleaning
    - Batch processing capabilities
    - Performance monitoring
    """
    
    # Supported models with their configurations
    SUPPORTED_MODELS = {
        'bart-large-cnn': {
            'tokenizer_class': 'AutoTokenizer',
            'model_class': 'AutoModelForSeq2SeqLM',
            'max_input_length': 1024,
            'max_output_length': 142,
            'min_output_length': 30,
            'description': 'BART model fine-tuned on CNN/DailyMail dataset'
        },
        'facebook/bart-large-cnn': {
            'tokenizer_class': 'BartTokenizer',
            'model_class': 'BartForConditionalGeneration',
            'max_input_length': 1024,
            'max_output_length': 142,
            'min_output_length': 30,
            'description': 'Facebook BART model for news summarization'
        },
        'google/pegasus-xsum': {
            'tokenizer_class': 'PegasusTokenizer',
            'model_class': 'PegasusForConditionalGeneration',
            'max_input_length': 512,
            'max_output_length': 64,
            'min_output_length': 20,
            'description': 'Pegasus model fine-tuned on XSum dataset'
        },
        'google/pegasus-cnn_dailymail': {
            'tokenizer_class': 'PegasusTokenizer',
            'model_class': 'PegasusForConditionalGeneration',
            'max_input_length': 1024,
            'max_output_length': 128,
            'min_output_length': 32,
            'description': 'Pegasus model fine-tuned on CNN/DailyMail'
        },
        't5-small': {
            'tokenizer_class': 'T5Tokenizer',
            'model_class': 'T5ForConditionalGeneration',
            'max_input_length': 512,
            'max_output_length': 150,
            'min_output_length': 25,
            'description': 'T5-small model for text summarization'
        },
        't5-base': {
            'tokenizer_class': 'T5Tokenizer',
            'model_class': 'T5ForConditionalGeneration',
            'max_input_length': 512,
            'max_output_length': 150,
            'min_output_length': 25,
            'description': 'T5-base model for text summarization'
        }
    }
    
    def __init__(self, model_name: str = 'facebook/bart-large-cnn',
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None):
        """
        Initialize EmailSummarizer.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run model on ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache downloaded models
            min_length: Minimum summary length (overrides model default)
            max_length: Maximum summary length (overrides model default)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ModelNotAvailableError(
                "HuggingFace transformers not available. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Validate model
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} not in supported list. Attempting to load anyway.")
            # Use default configuration for unknown models
            self.model_config = {
                'tokenizer_class': 'AutoTokenizer',
                'model_class': 'AutoModelForSeq2SeqLM',
                'max_input_length': 1024,
                'max_output_length': 150,
                'min_output_length': 30,
                'description': f'Custom model: {model_name}'
            }
        else:
            self.model_config = self.SUPPORTED_MODELS[model_name].copy()
        
        # Override length parameters if provided
        if min_length is not None:
            self.model_config['min_output_length'] = min_length
        if max_length is not None:
            self.model_config['max_output_length'] = max_length
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Model components (loaded lazily)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._model_loaded = False
        
        logger.info(f"Initialized EmailSummarizer with model: {model_name} on device: {self.device}")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Load tokenizer
            if self.model_config['tokenizer_class'] == 'AutoTokenizer':
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir
                )
            elif self.model_config['tokenizer_class'] == 'BartTokenizer':
                self.tokenizer = BartTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.model_config['tokenizer_class'] == 'PegasusTokenizer':
                self.tokenizer = PegasusTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.model_config['tokenizer_class'] == 'T5Tokenizer':
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            
            # Load model
            if self.model_config['model_class'] == 'AutoModelForSeq2SeqLM':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.model_config['model_class'] == 'BartForConditionalGeneration':
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.model_config['model_class'] == 'PegasusForConditionalGeneration':
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.model_config['model_class'] == 'T5ForConditionalGeneration':
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            
            # Move model to device
            self.model.to(self.device)
            
            # Create pipeline
            task = "summarization"
            if "t5" in self.model_name.lower():
                task = "text2text-generation"
            
            self.pipeline = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
            
            self._model_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise ModelNotAvailableError(f"Failed to load model: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess email text for summarization.
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove email headers and quoted text
        text = re.sub(r'^(From|To|Subject|Date|CC|BCC):.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>.*?$', '', text, flags=re.MULTILINE)  # Remove quoted lines
        text = re.sub(r'On .* wrote:', '', text)  # Remove "On ... wrote:" lines
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove email signatures (common patterns)
        text = re.sub(r'\n--\s*\n.*', '', text, flags=re.DOTALL)
        text = re.sub(r'\nBest regards.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\nSincerely.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\nThanks.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        # Remove very short lines (likely formatting artifacts)
        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
        text = '\n'.join(filtered_lines)
        
        return text.strip()
    
    def postprocess_summary(self, summary: str) -> str:
        """
        Post-process generated summary for better readability.
        
        Args:
            summary: Raw generated summary
            
        Returns:
            Cleaned and formatted summary
        """
        if not isinstance(summary, str):
            summary = str(summary)
        
        # Remove any remaining special tokens
        summary = re.sub(r'<[^>]+>', '', summary)
        summary = re.sub(r'\[.*?\]', '', summary)
        
        # Clean up punctuation
        summary = re.sub(r'\s+([.!?,:;])', r'\1', summary)  # Remove space before punctuation
        summary = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', summary)  # Ensure space after sentence end
        
        # Capitalize first letter
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure summary ends with proper punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Remove redundant phrases
        summary = re.sub(r'\bthe email\b', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'\bthis message\b', '', summary, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary
    
    def _prepare_input_text(self, text: str) -> str:
        """
        Prepare input text for specific model requirements.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Model-ready input text
        """
        # T5 models require specific prefixes
        if 't5' in self.model_name.lower():
            text = f"summarize: {text}"
        
        return text
    
    def _validate_text_length(self, text: str) -> None:
        """
        Validate that text is suitable for summarization.
        
        Args:
            text: Text to validate
            
        Raises:
            TextTooShortError: If text is too short to summarize
        """
        # Check minimum word count
        words = text.split()
        if len(words) < 20:
            raise TextTooShortError(f"Text too short for summarization: {len(words)} words (minimum: 20)")
        
        # Check if text is already very concise
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 2:
            raise TextTooShortError("Text already very concise (2 or fewer sentences)")
    
    def summarize(self, text: str, 
                  max_length: Optional[int] = None,
                  min_length: Optional[int] = None,
                  do_sample: bool = False,
                  temperature: float = 1.0,
                  top_p: float = 1.0,
                  repetition_penalty: float = 1.0) -> SummaryResult:
        """
        Generate a summary for the given text.
        
        Args:
            text: Email text to summarize
            max_length: Maximum summary length (overrides default)
            min_length: Minimum summary length (overrides default)
            do_sample: Whether to use sampling for generation
            temperature: Sampling temperature (if do_sample=True)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repetitive text
            
        Returns:
            SummaryResult object with summary and metadata
        """
        if not text or not text.strip():
            raise EmailSummarizerError("Input text is empty")
        
        # Load model if not already loaded
        self._load_model()
        
        start_time = time.time()
        original_text = text
        original_length = len(text)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Validate text length
        self._validate_text_length(processed_text)
        
        # Prepare input for model
        input_text = self._prepare_input_text(processed_text)
        
        # Set generation parameters
        generation_params = {
            'max_length': max_length or self.model_config['max_output_length'],
            'min_length': min_length or self.model_config['min_output_length'],
            'do_sample': do_sample,
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'early_stopping': True,
            'no_repeat_ngram_size': 2
        }
        
        # Handle model-specific parameters
        if 't5' in self.model_name.lower():
            # T5 models use different parameter names
            generation_params.pop('early_stopping', None)
        
        try:
            # Generate summary
            if 't5' in self.model_name.lower():
                # T5 uses text2text-generation pipeline
                result = self.pipeline(
                    input_text,
                    max_length=generation_params['max_length'],
                    min_length=generation_params['min_length'],
                    do_sample=generation_params['do_sample'],
                    temperature=generation_params['temperature'] if do_sample else None,
                    repetition_penalty=generation_params['repetition_penalty']
                )
                summary = result[0]['generated_text']
            else:
                # BART and Pegasus use summarization pipeline
                result = self.pipeline(
                    input_text,
                    **generation_params
                )
                summary = result[0]['summary_text']
            
            # Post-process summary
            summary = self.postprocess_summary(summary)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            # Create result
            result = SummaryResult(
                original_text=original_text,
                summary=summary,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                model_used=self.model_name
            )
            
            logger.info(f"Generated summary in {processing_time:.3f}s, "
                       f"compression: {compression_ratio:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise EmailSummarizerError(f"Failed to generate summary: {e}")
    
    def summarize_batch(self, texts: List[str], 
                       max_length: Optional[int] = None,
                       min_length: Optional[int] = None,
                       batch_size: int = 4) -> List[SummaryResult]:
        """
        Generate summaries for multiple texts efficiently.
        
        Args:
            texts: List of email texts to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of SummaryResult objects
        """
        if not texts:
            return []
        
        # Load model if not already loaded
        self._load_model()
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                try:
                    result = self.summarize(
                        text,
                        max_length=max_length,
                        min_length=min_length
                    )
                    batch_results.append(result)
                except (TextTooShortError, EmailSummarizerError) as e:
                    logger.warning(f"Skipping text due to error: {e}")
                    # Create a result with the original text as summary
                    batch_results.append(SummaryResult(
                        original_text=text,
                        summary=text[:100] + "..." if len(text) > 100 else text,
                        original_length=len(text),
                        summary_length=min(len(text), 100),
                        compression_ratio=1.0 if len(text) <= 100 else 100 / len(text),
                        processing_time=0.0,
                        model_used=self.model_name
                    ))
            
            results.extend(batch_results)
            
            # Log progress
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self._model_loaded,
            'config': self.model_config.copy(),
            'cache_dir': self.cache_dir
        }
        
        if self._model_loaded and hasattr(self.model, 'config'):
            info['model_parameters'] = self.model.num_parameters()
            info['model_config'] = self.model.config.to_dict()
        
        return info
    
    def estimate_processing_time(self, text_length: int) -> float:
        """
        Estimate processing time based on text length.
        
        Args:
            text_length: Length of input text in characters
            
        Returns:
            Estimated processing time in seconds
        """
        # Rough estimates based on typical performance
        if self.device == 'cuda':
            # GPU processing
            base_time = 0.5
            time_per_char = 0.0001
        else:
            # CPU processing
            base_time = 2.0
            time_per_char = 0.0005
        
        return base_time + (text_length * time_per_char)
    
    def supports_batch_processing(self) -> bool:
        """Check if the current model supports efficient batch processing."""
        return self._model_loaded and hasattr(self.pipeline, 'batch_size')
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, str]:
        """
        Get list of supported models with descriptions.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        return {
            model: config['description'] 
            for model, config in cls.SUPPORTED_MODELS.items()
        }
    
    @classmethod
    def get_recommended_model(cls, use_case: str = 'general') -> str:
        """
        Get recommended model for specific use case.
        
        Args:
            use_case: 'general', 'fast', 'quality', 'long', 'short'
            
        Returns:
            Recommended model name
        """
        recommendations = {
            'general': 'facebook/bart-large-cnn',
            'fast': 't5-small',
            'quality': 'google/pegasus-cnn_dailymail',
            'long': 'facebook/bart-large-cnn',
            'short': 'google/pegasus-xsum'
        }
        
        return recommendations.get(use_case, 'facebook/bart-large-cnn')
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            # Move model to CPU to free GPU memory
            try:
                if self.device == 'cuda':
                    self.model.cpu()
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore cleanup errors