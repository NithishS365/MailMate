"""
TextToSpeech module for MailMate - AI-powered text-to-speech conversion.

This module provides text-to-speech functionality for email summaries, notifications,
and other text content using multiple TTS engines:
- gTTS (Google Text-to-Speech) for high-quality online synthesis
- pyttsx3 for offline text-to-speech capabilities
- Fallback mechanisms and comprehensive error handling
"""

import io
import logging
import os
import tempfile
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Optional imports with graceful fallbacks
try:
    import gtts
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    warnings.warn("gTTS not available. Install with: pip install gtts")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    warnings.warn("pyttsx3 not available. Install with: pip install pyttsx3")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    warnings.warn("pygame not available. Install with: pip install pygame (optional for audio playback)")

# Configure logging
logger = logging.getLogger(__name__)

class TTSEngine(Enum):
    """Text-to-speech engine types."""
    GTTS = "gtts"
    PYTTSX3 = "pyttsx3"
    AUTO = "auto"

class AudioFormat(Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"

@dataclass
class TTSResult:
    """Result of text-to-speech conversion."""
    success: bool
    audio_file_path: Optional[str]
    text_processed: str
    engine_used: str
    processing_time: float
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class VoiceInfo:
    """Information about available voices."""
    voice_id: str
    name: str
    language: str
    gender: Optional[str] = None
    age: Optional[str] = None

class TextToSpeechError(Exception):
    """Base exception for TextToSpeech errors."""
    pass

class EngineNotAvailableError(TextToSpeechError):
    """Exception raised when requested TTS engine is not available."""
    pass

class AudioConversionError(TextToSpeechError):
    """Exception raised when audio conversion fails."""
    pass

class FileOutputError(TextToSpeechError):
    """Exception raised when file output operations fail."""
    pass

class TextToSpeech:
    """
    Comprehensive text-to-speech converter supporting multiple engines.
    
    Features:
    - Multiple TTS engines (gTTS, pyttsx3)
    - Automatic engine selection and fallback
    - Audio format conversion
    - Voice customization
    - Batch processing
    - Audio playback capabilities
    - Comprehensive error handling
    """
    
    def __init__(self, 
                 engine: TTSEngine = TTSEngine.AUTO,
                 language: str = 'en',
                 voice_speed: int = 150,
                 voice_volume: float = 0.9):
        """
        Initialize TextToSpeech converter.
        
        Args:
            engine: TTS engine to use (AUTO, GTTS, PYTTSX3)
            language: Language code (e.g., 'en', 'es', 'fr')
            voice_speed: Speech rate (words per minute, for pyttsx3)
            voice_volume: Voice volume (0.0 to 1.0, for pyttsx3)
        """
        self.engine_type = engine
        self.language = language
        self.voice_speed = voice_speed
        self.voice_volume = voice_volume
        
        # Engine instances
        self._gtts_engine = None
        self._pyttsx3_engine = None
        
        # Configuration
        self.supported_languages = self._get_supported_languages()
        self.available_engines = self._check_available_engines()
        
        # Select and initialize engine
        self.current_engine = self._select_engine()
        self._initialize_engine()
        
        logger.info(f"TextToSpeech initialized with engine: {self.current_engine}")
    
    def _check_available_engines(self) -> List[str]:
        """Check which TTS engines are available."""
        engines = []
        if GTTS_AVAILABLE:
            engines.append(TTSEngine.GTTS.value)
        if PYTTSX3_AVAILABLE:
            engines.append(TTSEngine.PYTTSX3.value)
        return engines
    
    def _select_engine(self) -> TTSEngine:
        """Select the best available TTS engine."""
        if self.engine_type == TTSEngine.AUTO:
            # Prefer gTTS for quality, fallback to pyttsx3 for offline
            if GTTS_AVAILABLE:
                return TTSEngine.GTTS
            elif PYTTSX3_AVAILABLE:
                return TTSEngine.PYTTSX3
            else:
                raise EngineNotAvailableError("No TTS engines available. Install gtts or pyttsx3.")
        
        elif self.engine_type == TTSEngine.GTTS:
            if not GTTS_AVAILABLE:
                raise EngineNotAvailableError("gTTS not available. Install with: pip install gtts")
            return TTSEngine.GTTS
        
        elif self.engine_type == TTSEngine.PYTTSX3:
            if not PYTTSX3_AVAILABLE:
                raise EngineNotAvailableError("pyttsx3 not available. Install with: pip install pyttsx3")
            return TTSEngine.PYTTSX3
        
        else:
            raise ValueError(f"Unknown engine type: {self.engine_type}")
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine."""
        if self.current_engine == TTSEngine.GTTS:
            self._initialize_gtts()
        elif self.current_engine == TTSEngine.PYTTSX3:
            self._initialize_pyttsx3()
    
    def _initialize_gtts(self):
        """Initialize gTTS engine."""
        try:
            # Test gTTS availability by creating a minimal instance
            test_tts = gTTS(text="test", lang=self.language, slow=False)
            self._gtts_engine = "initialized"
            logger.info("gTTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize gTTS: {e}")
            raise EngineNotAvailableError(f"gTTS initialization failed: {e}")
    
    def _initialize_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        try:
            self._pyttsx3_engine = pyttsx3.init()
            
            # Configure voice properties
            self._pyttsx3_engine.setProperty('rate', self.voice_speed)
            self._pyttsx3_engine.setProperty('volume', self.voice_volume)
            
            # Set voice if language preference is available
            voices = self._pyttsx3_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if self.language.lower() in voice.id.lower():
                        self._pyttsx3_engine.setProperty('voice', voice.id)
                        break
            
            logger.info("pyttsx3 engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise EngineNotAvailableError(f"pyttsx3 initialization failed: {e}")
    
    def _get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for different engines."""
        # Common language codes
        common_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
        # gTTS supports many more languages
        if GTTS_AVAILABLE:
            try:
                from gtts.lang import tts_langs
                gtts_languages = tts_langs()
                common_languages.update(gtts_languages)
            except ImportError:
                pass
        
        return common_languages
    
    def convert_text_to_speech(self, 
                             text: str, 
                             output_path: Union[str, Path],
                             audio_format: AudioFormat = AudioFormat.MP3,
                             language: Optional[str] = None) -> TTSResult:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path where audio file will be saved
            audio_format: Output audio format (MP3 or WAV)
            language: Language override for this conversion
            
        Returns:
            TTSResult with conversion details
        """
        start_time = time.time()
        output_path = Path(output_path)
        lang = language or self.language
        
        logger.info(f"Converting text to speech: {len(text)} characters")
        
        try:
            # Validate input
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert based on engine
            if self.current_engine == TTSEngine.GTTS:
                success = self._convert_with_gtts(processed_text, output_path, lang, audio_format)
            elif self.current_engine == TTSEngine.PYTTSX3:
                success = self._convert_with_pyttsx3(processed_text, output_path, audio_format)
            else:
                raise EngineNotAvailableError(f"Engine {self.current_engine} not supported")
            
            processing_time = time.time() - start_time
            
            if success and output_path.exists():
                file_size = output_path.stat().st_size
                duration = self._estimate_audio_duration(processed_text)
                
                return TTSResult(
                    success=True,
                    audio_file_path=str(output_path),
                    text_processed=processed_text,
                    engine_used=self.current_engine.value,
                    processing_time=processing_time,
                    file_size_bytes=file_size,
                    duration_seconds=duration
                )
            else:
                return TTSResult(
                    success=False,
                    audio_file_path=None,
                    text_processed=processed_text,
                    engine_used=self.current_engine.value,
                    processing_time=processing_time,
                    error_message="Audio conversion failed"
                )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Text-to-speech conversion failed: {e}")
            
            return TTSResult(
                success=False,
                audio_file_path=None,
                text_processed=text,
                engine_used=self.current_engine.value,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _convert_with_gtts(self, 
                          text: str, 
                          output_path: Path, 
                          language: str,
                          audio_format: AudioFormat) -> bool:
        """Convert text using gTTS engine."""
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            if audio_format == AudioFormat.MP3:
                # Direct MP3 output
                tts.save(str(output_path))
            elif audio_format == AudioFormat.WAV:
                # gTTS outputs MP3, need conversion for WAV
                temp_mp3 = output_path.with_suffix('.tmp.mp3')
                tts.save(str(temp_mp3))
                
                # Convert MP3 to WAV (would need additional library like pydub)
                self._convert_audio_format(temp_mp3, output_path, AudioFormat.WAV)
                temp_mp3.unlink()  # Clean up temporary file
            
            return True
            
        except Exception as e:
            logger.error(f"gTTS conversion failed: {e}")
            return False
    
    def _convert_with_pyttsx3(self, 
                            text: str, 
                            output_path: Path, 
                            audio_format: AudioFormat) -> bool:
        """Convert text using pyttsx3 engine."""
        try:
            if audio_format == AudioFormat.WAV:
                # pyttsx3 can save directly to WAV
                self._pyttsx3_engine.save_to_file(text, str(output_path))
                self._pyttsx3_engine.runAndWait()
            else:
                # For MP3, save as WAV first then convert
                temp_wav = output_path.with_suffix('.tmp.wav')
                self._pyttsx3_engine.save_to_file(text, str(temp_wav))
                self._pyttsx3_engine.runAndWait()
                
                # Convert WAV to MP3 (would need additional library)
                self._convert_audio_format(temp_wav, output_path, AudioFormat.MP3)
                temp_wav.unlink()  # Clean up temporary file
            
            return True
            
        except Exception as e:
            logger.error(f"pyttsx3 conversion failed: {e}")
            return False
    
    def _convert_audio_format(self, 
                            input_path: Path, 
                            output_path: Path, 
                            target_format: AudioFormat):
        """Convert audio between formats (placeholder for audio conversion)."""
        # This is a placeholder - in a real implementation, you'd use
        # libraries like pydub, ffmpeg-python, or similar
        logger.warning(f"Audio format conversion not implemented: {input_path} -> {output_path}")
        
        # For now, just copy the file
        import shutil
        shutil.copy2(input_path, output_path)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better TTS output."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Expand common abbreviations for better pronunciation
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'e.g.': 'for example',
            'i.e.': 'that is',
            '&': 'and'
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Limit text length for better processing
        max_length = 5000  # Most TTS engines have limits
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    def _estimate_audio_duration(self, text: str) -> float:
        """Estimate audio duration based on text length."""
        # Average speaking rate: ~150 words per minute
        words = len(text.split())
        duration_minutes = words / 150
        return duration_minutes * 60  # Convert to seconds
    
    def convert_batch(self, 
                     texts: List[str], 
                     output_directory: Union[str, Path],
                     filename_prefix: str = "audio",
                     audio_format: AudioFormat = AudioFormat.MP3) -> List[TTSResult]:
        """
        Convert multiple texts to speech files.
        
        Args:
            texts: List of texts to convert
            output_directory: Directory to save audio files
            filename_prefix: Prefix for generated filenames
            audio_format: Output audio format
            
        Returns:
            List of TTSResult objects
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, text in enumerate(texts, 1):
            filename = f"{filename_prefix}_{i:03d}.{audio_format.value}"
            output_path = output_dir / filename
            
            result = self.convert_text_to_speech(text, output_path, audio_format)
            results.append(result)
            
            logger.info(f"Batch processing: {i}/{len(texts)} completed")
        
        return results
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """Get list of available voices for current engine."""
        voices = []
        
        if self.current_engine == TTSEngine.PYTTSX3 and self._pyttsx3_engine:
            try:
                pyttsx3_voices = self._pyttsx3_engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append(VoiceInfo(
                        voice_id=voice.id,
                        name=voice.name,
                        language=getattr(voice, 'languages', ['unknown'])[0] if hasattr(voice, 'languages') else 'unknown',
                        gender=getattr(voice, 'gender', None),
                        age=getattr(voice, 'age', None)
                    ))
            except Exception as e:
                logger.error(f"Failed to get pyttsx3 voices: {e}")
        
        elif self.current_engine == TTSEngine.GTTS:
            # gTTS doesn't have voice selection, but has language support
            for lang_code, lang_name in self.supported_languages.items():
                voices.append(VoiceInfo(
                    voice_id=lang_code,
                    name=f"gTTS {lang_name}",
                    language=lang_code
                ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Set the voice for TTS engine.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Success status
        """
        try:
            if self.current_engine == TTSEngine.PYTTSX3 and self._pyttsx3_engine:
                self._pyttsx3_engine.setProperty('voice', voice_id)
                return True
            elif self.current_engine == TTSEngine.GTTS:
                # For gTTS, voice_id should be language code
                if voice_id in self.supported_languages:
                    self.language = voice_id
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set voice {voice_id}: {e}")
            return False
    
    def play_audio(self, audio_file_path: Union[str, Path]) -> bool:
        """
        Play audio file (requires pygame).
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Success status
        """
        if not PYGAME_AVAILABLE:
            logger.warning("pygame not available for audio playback")
            return False
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(str(audio_file_path))
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about current TTS engine configuration."""
        return {
            'current_engine': self.current_engine.value,
            'available_engines': self.available_engines,
            'language': self.language,
            'voice_speed': self.voice_speed,
            'voice_volume': self.voice_volume,
            'supported_languages': list(self.supported_languages.keys()),
            'gtts_available': GTTS_AVAILABLE,
            'pyttsx3_available': PYTTSX3_AVAILABLE,
            'pygame_available': PYGAME_AVAILABLE
        }
    
    @staticmethod
    def list_supported_engines() -> Dict[str, Dict[str, Any]]:
        """List all supported TTS engines and their capabilities."""
        engines = {
            'gtts': {
                'name': 'Google Text-to-Speech',
                'available': GTTS_AVAILABLE,
                'online_required': True,
                'quality': 'High',
                'languages': 'Many',
                'install_command': 'pip install gtts'
            },
            'pyttsx3': {
                'name': 'pyttsx3 Offline TTS',
                'available': PYTTSX3_AVAILABLE,
                'online_required': False,
                'quality': 'Medium',
                'languages': 'System dependent',
                'install_command': 'pip install pyttsx3'
            }
        }
        return engines
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            if self._pyttsx3_engine:
                # pyttsx3 cleanup if needed
                pass
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Convenience functions
def text_to_speech_simple(text: str, 
                         output_path: Union[str, Path],
                         engine: TTSEngine = TTSEngine.AUTO,
                         language: str = 'en') -> TTSResult:
    """
    Simple text-to-speech conversion function.
    
    Args:
        text: Text to convert
        output_path: Output file path
        engine: TTS engine to use
        language: Language code
        
    Returns:
        TTSResult with conversion details
    """
    with TextToSpeech(engine=engine, language=language) as tts:
        return tts.convert_text_to_speech(text, output_path)

def convert_email_summary_to_audio(summary_text: str, 
                                  subject: str,
                                  output_directory: Union[str, Path]) -> TTSResult:
    """
    Convert email summary to audio with enhanced formatting.
    
    Args:
        summary_text: Email summary text
        subject: Email subject line
        output_directory: Directory to save audio file
        
    Returns:
        TTSResult with conversion details
    """
    # Create formatted text for TTS
    formatted_text = f"Email summary: {subject}. {summary_text}"
    
    # Generate filename from subject
    safe_filename = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_filename = safe_filename[:50]  # Limit length
    filename = f"email_summary_{safe_filename}.mp3"
    
    output_path = Path(output_directory) / filename
    
    return text_to_speech_simple(formatted_text, output_path)