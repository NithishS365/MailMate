"""
Complete MailMate Integration Example with TextToSpeech

This example demonstrates the full MailMate AI Email Management System including:
- EmailDataLoader for data management
- EmailClassifier for categorization
- EmailSummarizer for content summarization  
- TextToSpeech for audio conversion
- Complete end-to-end email processing pipeline
"""

import os
import sys
from pathlib import Path
import time

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from email_loader import EmailDataLoader
from email_classifier import EmailClassifier
from email_summarizer import EmailSummarizer
from text_to_speech import TextToSpeech, TTSEngine, convert_email_summary_to_audio

def demonstrate_complete_integration_with_audio():
    """Demonstrate complete MailMate system with TextToSpeech integration."""
    print("\n" + "="*80)
    print("MAILMATE COMPLETE INTEGRATION WITH TEXT-TO-SPEECH")
    print("="*80)
    
    # Initialize all components
    print("Initializing MailMate components...")
    
    try:
        loader = EmailDataLoader()
        classifier = EmailClassifier()
        summarizer = EmailSummarizer()
        tts = TextToSpeech(engine=TTSEngine.AUTO)
        
        print("✓ All components initialized successfully")
        
        # Create output directories
        audio_dir = Path("audio_outputs/integration")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Generate and classify emails
        print(f"\nPhase 1: Email Generation and Classification")
        print("-" * 50)
        
        emails = loader.generate_synthetic_emails(count=200)
        print(f"Generated {len(emails)} synthetic emails")
        
        # Train classifier
        print("Training email classifier...")
        train_result = classifier.train(emails)
        print(f"✓ Classifier trained with {train_result.accuracy:.1%} accuracy")
        
        # Phase 2: Process sample emails with audio
        print(f"\nPhase 2: Email Processing with Audio Generation")
        print("-" * 50)
        
        # Sample emails for processing
        sample_emails = [
            {
                "subject": "Urgent: Server Maintenance Tonight",
                "content": "We will be performing critical server maintenance tonight from 11 PM to 3 AM. Please save all work and log out before 11 PM. The system will be completely unavailable during this time. Contact IT support if you have any questions or concerns.",
                "sender": "it-support@company.com"
            },
            {
                "subject": "Q4 Budget Review Meeting",
                "content": "The quarterly budget review meeting is scheduled for Friday at 2 PM in Conference Room A. Please bring your department's budget proposals and Q3 expense reports. We'll be discussing budget allocations for the next quarter and planning for the upcoming fiscal year.",
                "sender": "finance@company.com"
            },
            {
                "subject": "Welcome to the Team!",
                "content": "We're excited to welcome you to our team! Your first day is Monday, and we've prepared an orientation schedule. You'll meet with HR at 9 AM, then join your team for introductions. Lunch will be provided, and we'll give you a tour of the facilities.",
                "sender": "hr@company.com"
            }
        ]
        
        processing_results = []
        
        for i, email in enumerate(sample_emails, 1):
            print(f"\nProcessing Email {i}: {email['subject']}")
            
            try:
                # Step 1: Classify email
                classification = classifier.predict(email["content"])
                print(f"  Classification: {classification.predicted_category} ({classification.confidence:.1%} confidence)")
                
                # Step 2: Summarize email
                summary_result = summarizer.summarize_email(email["content"])
                
                if summary_result.success:
                    print(f"  Summary: {summary_result.summary[:100]}...")
                    
                    # Step 3: Convert summary to audio
                    audio_result = convert_email_summary_to_audio(
                        summary_text=summary_result.summary,
                        subject=email["subject"],
                        output_directory=audio_dir
                    )
                    
                    if audio_result.success:
                        audio_file = Path(audio_result.audio_file_path).name
                        print(f"  ✓ Audio generated: {audio_file}")
                        print(f"    Processing time: {audio_result.processing_time:.2f}s")
                        print(f"    File size: {audio_result.file_size_bytes:,} bytes")
                        print(f"    Duration: {audio_result.duration_seconds:.1f}s")
                    else:
                        print(f"  ✗ Audio generation failed: {audio_result.error_message}")
                    
                    processing_results.append({
                        "email": email,
                        "classification": classification,
                        "summary": summary_result,
                        "audio": audio_result
                    })
                else:
                    print(f"  ✗ Summarization failed: {summary_result.error_message}")
                    
                    # Still try to create audio from original content
                    audio_result = convert_email_summary_to_audio(
                        summary_text=email["content"][:200] + "...",
                        subject=email["subject"],
                        output_directory=audio_dir
                    )
                    
                    if audio_result.success:
                        print(f"  ✓ Audio from original content: {Path(audio_result.audio_file_path).name}")
                    
            except Exception as e:
                print(f"  ✗ Processing failed: {e}")
        
        # Phase 3: Audio notifications and batch processing
        print(f"\nPhase 3: Audio Notifications and Batch Processing")
        print("-" * 50)
        
        # Create category-based notifications
        categories = ["work", "personal", "finance", "shopping", "travel"]
        notification_texts = []
        
        for category in categories:
            category_emails = [r for r in processing_results if r["classification"].predicted_category.lower() == category]
            if category_emails:
                count = len(category_emails)
                avg_confidence = sum(r["classification"].confidence for r in category_emails) / count
                notification = f"You have {count} new {category} emails with average confidence {avg_confidence:.1%}"
                notification_texts.append(notification)
        
        if notification_texts:
            print(f"Generating {len(notification_texts)} category notifications...")
            
            batch_results = tts.convert_batch(
                texts=notification_texts,
                output_directory=audio_dir / "notifications",
                filename_prefix="category_notification",
                audio_format=tts.AudioFormat.MP3 if hasattr(tts, 'AudioFormat') else "mp3"
            )
            
            successful_notifications = sum(1 for r in batch_results if r.success)
            print(f"✓ Generated {successful_notifications}/{len(batch_results)} notification audio files")
        
        # Phase 4: System status and summary
        print(f"\nPhase 4: System Status and Final Summary")
        print("-" * 50)
        
        # Generate system status audio
        total_emails_processed = len(processing_results)
        successful_summaries = sum(1 for r in processing_results if r["summary"].success)
        successful_audio = sum(1 for r in processing_results if r["audio"].success)
        
        status_message = f"""
        MailMate system status report: Processing completed successfully. 
        {total_emails_processed} emails were processed. 
        {successful_summaries} summaries were generated. 
        {successful_audio} audio files were created. 
        System is operating normally.
        """
        
        status_audio = tts.convert_text_to_speech(
            text=status_message.strip(),
            output_path=audio_dir / "system_status.mp3"
        )
        
        if status_audio.success:
            print(f"✓ System status audio generated: {status_audio.processing_time:.2f}s")
        
        # Show engine information
        engine_info = tts.get_engine_info()
        print(f"\nTTS Engine Information:")
        print(f"- Current engine: {engine_info['current_engine']}")
        print(f"- Available engines: {', '.join(engine_info['available_engines'])}")
        print(f"- Language: {engine_info['language']}")
        
        # Final statistics
        print(f"\n" + "="*80)
        print("INTEGRATION SUMMARY")
        print("="*80)
        print(f"📧 Emails processed: {total_emails_processed}")
        print(f"📋 Summaries generated: {successful_summaries}")
        print(f"🔊 Audio files created: {successful_audio}")
        print(f"📁 Audio output directory: {audio_dir}")
        
        # List generated audio files
        audio_files = list(audio_dir.rglob("*.mp3"))
        if audio_files:
            print(f"\n🎵 Generated Audio Files ({len(audio_files)}):")
            for audio_file in audio_files:
                relative_path = audio_file.relative_to(audio_dir)
                file_size = audio_file.stat().st_size
                print(f"  - {relative_path} ({file_size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_tts_features():
    """Demonstrate specific TextToSpeech features."""
    print("\n" + "="*80)
    print("TEXT-TO-SPEECH FEATURE DEMONSTRATION")
    print("="*80)
    
    try:
        tts = TextToSpeech()
        
        # Show available engines
        engines = TextToSpeech.list_supported_engines()
        print("Available TTS Engines:")
        for engine_id, info in engines.items():
            status = "✓ Available" if info['available'] else "✗ Not Available"
            print(f"  {info['name']}: {status}")
            print(f"    Quality: {info['quality']}, Online Required: {info['online_required']}")
        
        # Voice information
        voices = tts.get_available_voices()
        print(f"\nAvailable Voices: {len(voices)}")
        for voice in voices[:3]:  # Show first 3 voices
            print(f"  - {voice.name} ({voice.language})")
        
        # Test different text types
        test_cases = [
            {
                "name": "Email subject line",
                "text": "Urgent: Server maintenance scheduled for tonight"
            },
            {
                "name": "Email summary",
                "text": "The server will be down for maintenance from 11 PM to 3 AM. Please save your work."
            },
            {
                "name": "Notification",
                "text": "You have 5 new work emails with high priority."
            }
        ]
        
        print(f"\nTesting different text types:")
        audio_dir = Path("audio_outputs/tts_features")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"  {i}. {test_case['name']}")
            
            result = tts.convert_text_to_speech(
                text=test_case["text"],
                output_path=audio_dir / f"test_{i}_{test_case['name'].replace(' ', '_')}.mp3"
            )
            
            if result.success:
                print(f"     ✓ Generated in {result.processing_time:.2f}s")
            else:
                print(f"     ✗ Failed: {result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"TTS feature demonstration failed: {e}")
        return False

def main():
    """Run complete MailMate integration demonstration with TextToSpeech."""
    print("MailMate AI Email Management System")
    print("Complete Integration with Text-to-Speech")
    print("=" * 80)
    
    try:
        # Run main integration
        integration_success = demonstrate_complete_integration_with_audio()
        
        # Run TTS feature demo
        tts_success = demonstrate_tts_features()
        
        if integration_success and tts_success:
            print(f"\n🎉 All demonstrations completed successfully!")
            print(f"Check the 'audio_outputs' directory for generated audio files.")
        else:
            print(f"\n⚠️ Some demonstrations had issues. Check the output above for details.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()