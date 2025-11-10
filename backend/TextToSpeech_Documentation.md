# TextToSpeech Module Documentation

## Overview

The TextToSpeech module provides comprehensive text-to-speech capabilities for the MailMate AI Email Management System. It supports multiple TTS engines, audio formats, and includes robust error handling for converting email summaries, notifications, and other text content into high-quality audio files.

## Features

### Core Capabilities
- **Multiple TTS Engines**: Support for gTTS (Google Text-to-Speech) and pyttsx3 (offline TTS)
- **Automatic Engine Selection**: Intelligent fallback between available engines
- **Audio Format Support**: MP3 and WAV output formats
- **Batch Processing**: Convert multiple texts efficiently
- **Voice Customization**: Speed, volume, and voice selection
- **Audio Playback**: Built-in playback capabilities (with pygame)
- **Comprehensive Error Handling**: Graceful handling of network issues, missing dependencies, and invalid inputs

### Email Integration
- **Email Summary Audio**: Convert email summaries to audio with formatted announcements
- **Subject Line Integration**: Include email subjects in audio output
- **Batch Email Processing**: Process multiple emails in sequence
- **Filename Generation**: Automatic, safe filename generation from email subjects

## Installation

### Required Dependencies
```bash
pip install gtts>=2.3.0          # Google Text-to-Speech (online)
pip install pyttsx3>=2.90        # Offline text-to-speech
```

### Optional Dependencies
```bash
pip install pygame>=2.5.0        # For audio playback functionality
pip install pydub>=0.25.0        # For advanced audio format conversion
```

### Installation Verification
```python
from backend.text_to_speech import TextToSpeech

# Check available engines
engines = TextToSpeech.list_supported_engines()
for engine_id, info in engines.items():
    print(f"{info['name']}: {'Available' if info['available'] else 'Not Available'}")
```

## Quick Start

### Basic Usage
```python
from backend.text_to_speech import text_to_speech_simple, TTSEngine

# Simple text-to-speech conversion
result = text_to_speech_simple(
    text="Hello, this is a test of the text-to-speech system.",
    output_path="output/hello.mp3",
    engine=TTSEngine.AUTO,  # Automatically select best available engine
    language='en'
)

print(f"Success: {result.success}")
print(f"Output file: {result.audio_file_path}")
```

### Email Summary Integration
```python
from backend.text_to_speech import convert_email_summary_to_audio

# Convert email summary to audio
result = convert_email_summary_to_audio(
    summary_text="The quarterly sales report shows a 15% increase in revenue.",
    subject="Q4 Sales Report - Action Required", 
    output_directory="audio_outputs"
)

if result.success:
    print(f"Audio saved to: {result.audio_file_path}")
```

## Advanced Usage

### Custom TTS Configuration
```python
from backend.text_to_speech import TextToSpeech, TTSEngine, AudioFormat

# Initialize with custom settings
tts = TextToSpeech(
    engine=TTSEngine.PYTTSX3,  # Force specific engine
    language='en',
    voice_speed=160,           # Words per minute
    voice_volume=0.8          # Volume level (0.0-1.0)
)

# Convert with custom format
result = tts.convert_text_to_speech(
    text="This is a custom configuration example.",
    output_path="output/custom.wav",
    audio_format=AudioFormat.WAV
)
```

### Batch Processing
```python
# Prepare multiple texts
email_summaries = [
    "Meeting scheduled for tomorrow at 2 PM.",
    "Project deadline extended by one week.",
    "New security protocols implemented."
]

# Convert all texts
results = tts.convert_batch(
    texts=email_summaries,
    output_directory="batch_output",
    filename_prefix="email_summary",
    audio_format=AudioFormat.MP3
)

# Check results
successful = sum(1 for r in results if r.success)
print(f"Successfully converted {successful}/{len(results)} texts")
```

### Voice Management
```python
# Get available voices
voices = tts.get_available_voices()
for voice in voices:
    print(f"Voice: {voice.name} (Language: {voice.language})")

# Set specific voice (for pyttsx3)
if voices:
    success = tts.set_voice(voices[0].voice_id)
    print(f"Voice set: {success}")
```

### Audio Playback
```python
# Play generated audio (requires pygame)
if result.success:
    playback_success = tts.play_audio(result.audio_file_path)
    print(f"Playback: {playback_success}")
```

## Engine Comparison

### gTTS (Google Text-to-Speech)
- **Quality**: High-quality, natural-sounding speech
- **Languages**: 100+ languages supported
- **Connection**: Requires internet connection
- **Formats**: MP3 output (native)
- **Speed**: Fast conversion for short texts
- **Limitations**: Rate limiting, requires Google services access

### pyttsx3 (Offline TTS)
- **Quality**: Good quality, system-dependent
- **Languages**: Limited to system-installed voices
- **Connection**: Works offline
- **Formats**: WAV output (native), can convert to MP3
- **Speed**: Fast, no network delays
- **Limitations**: Voice quality varies by system

### Automatic Selection Logic
1. If `TTSEngine.AUTO` is specified:
   - Prefer gTTS for better quality (if available and online)
   - Fallback to pyttsx3 for offline capability
   - Raise error if no engines available

## Configuration Options

### Engine Settings
```python
class TTSEngine(Enum):
    GTTS = "gtts"         # Google Text-to-Speech
    PYTTSX3 = "pyttsx3"   # Offline TTS engine
    AUTO = "auto"         # Automatic selection
```

### Audio Formats
```python
class AudioFormat(Enum):
    MP3 = "mp3"          # MPEG-1 Audio Layer 3
    WAV = "wav"          # Waveform Audio File Format
```

### Language Codes
Common language codes supported:
- `'en'` - English
- `'es'` - Spanish  
- `'fr'` - French
- `'de'` - German
- `'it'` - Italian
- `'pt'` - Portuguese
- `'ru'` - Russian
- `'ja'` - Japanese
- `'ko'` - Korean
- `'zh'` - Chinese
- `'ar'` - Arabic
- `'hi'` - Hindi

## Error Handling

### Exception Types
```python
# Base exception
TextToSpeechError

# Engine not available
EngineNotAvailableError

# Audio conversion failed
AudioConversionError

# File output issues
FileOutputError
```

### Error Handling Patterns
```python
try:
    result = tts.convert_text_to_speech(text, output_path)
    if not result.success:
        print(f"Conversion failed: {result.error_message}")
except EngineNotAvailableError as e:
    print(f"TTS engine not available: {e}")
except TextToSpeechError as e:
    print(f"TTS error: {e}")
```

### Graceful Degradation
```python
# Check engine availability before use
engines_info = TextToSpeech.list_supported_engines()

if engines_info['gtts']['available']:
    engine = TTSEngine.GTTS
elif engines_info['pyttsx3']['available']:  
    engine = TTSEngine.PYTTSX3
else:
    print("No TTS engines available - install gtts or pyttsx3")
    engine = None
```

## Integration with MailMate Components

### EmailSummarizer Integration
```python
from backend.email_summarizer import EmailSummarizer
from backend.text_to_speech import convert_email_summary_to_audio

# Generate summary and convert to audio
summarizer = EmailSummarizer()
summary_result = summarizer.summarize_email(email_content)

if summary_result.success:
    audio_result = convert_email_summary_to_audio(
        summary_text=summary_result.summary,
        subject=email.subject,
        output_directory="audio_summaries"
    )
```

### EmailClassifier Integration  
```python
from backend.email_classifier import EmailClassifier
from backend.text_to_speech import TextToSpeech

# Create audio notifications for classified emails
classifier = EmailClassifier()
tts = TextToSpeech()

classification = classifier.predict(email_content)
notification_text = f"New {classification.predicted_category} email received with {classification.confidence:.1%} confidence."

audio_result = tts.convert_text_to_speech(
    text=notification_text,
    output_path=f"notifications/{classification.predicted_category}_notification.mp3"
)
```

## Performance Considerations

### Text Length Optimization
- **Automatic Truncation**: Texts longer than 5,000 characters are automatically truncated
- **Preprocessing**: Removes excessive whitespace and expands abbreviations
- **Batch Processing**: Efficiently handles multiple texts with progress tracking

### File Management
- **Automatic Directory Creation**: Output directories are created automatically
- **Safe Filename Generation**: Special characters in filenames are handled safely
- **File Size Tracking**: Result objects include file size information

### Memory Usage
- **Streaming Processing**: Large texts are processed efficiently without loading entirely into memory
- **Temporary File Cleanup**: Temporary files are automatically cleaned up
- **Context Manager Support**: Use with `with` statements for automatic resource cleanup

## Testing

### Running Tests
```bash
# Run all TextToSpeech tests
python -m pytest tests/test_text_to_speech.py -v

# Run with coverage
python -m pytest tests/test_text_to_speech.py --cov=backend.text_to_speech --cov-report=html
```

### Test Coverage
- Basic TTS conversion functionality
- Engine selection and initialization
- Error handling and edge cases
- Audio format support
- Batch processing
- Voice management
- Integration scenarios

### Manual Testing
```bash
# Run the example script
cd backend
python text_to_speech_example.py
```

## Troubleshooting

### Common Issues

#### "No TTS engines available"
```bash
# Install missing dependencies
pip install gtts pyttsx3
```

#### "gTTS network error"
- Check internet connection
- Verify Google services are accessible
- Try using pyttsx3 as fallback

#### "pyttsx3 initialization failed"
- Check system TTS engines are installed
- On Windows: Ensure SAPI5 voices are available
- On Linux: Install espeak or festival
- On macOS: System voices should be available

#### "Audio file not created"
- Check output directory permissions
- Verify disk space availability
- Ensure output path is valid

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger('text_to_speech')
logger.setLevel(logging.DEBUG)
```

## API Reference

### Classes

#### TextToSpeech
Main text-to-speech converter class.

**Constructor:**
```python
TextToSpeech(engine=TTSEngine.AUTO, language='en', voice_speed=150, voice_volume=0.9)
```

**Methods:**
- `convert_text_to_speech(text, output_path, audio_format=AudioFormat.MP3, language=None) -> TTSResult`
- `convert_batch(texts, output_directory, filename_prefix="audio", audio_format=AudioFormat.MP3) -> List[TTSResult]`
- `get_available_voices() -> List[VoiceInfo]`
- `set_voice(voice_id) -> bool`
- `play_audio(audio_file_path) -> bool`
- `get_engine_info() -> Dict[str, Any]`

#### TTSResult
Result object containing conversion details.

**Properties:**
- `success: bool` - Conversion success status
- `audio_file_path: Optional[str]` - Path to generated audio file
- `text_processed: str` - Text that was processed
- `engine_used: str` - TTS engine that was used
- `processing_time: float` - Time taken for conversion
- `file_size_bytes: Optional[int]` - Size of generated file
- `duration_seconds: Optional[float]` - Estimated audio duration
- `error_message: Optional[str]` - Error message if failed

#### VoiceInfo
Information about available voices.

**Properties:**
- `voice_id: str` - Unique voice identifier
- `name: str` - Human-readable voice name
- `language: str` - Language code
- `gender: Optional[str]` - Voice gender
- `age: Optional[str]` - Voice age category

### Functions

#### text_to_speech_simple()
```python
text_to_speech_simple(text, output_path, engine=TTSEngine.AUTO, language='en') -> TTSResult
```

#### convert_email_summary_to_audio()
```python
convert_email_summary_to_audio(summary_text, subject, output_directory) -> TTSResult
```

## Examples

See `text_to_speech_example.py` for comprehensive usage examples covering:
- Basic text-to-speech conversion
- Advanced TTS features
- Email summary integration  
- Engine comparison
- Error handling scenarios

## License

This module is part of the MailMate AI Email Management System and follows the same licensing terms as the main project.