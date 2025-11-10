"""
MailMate TextToSpeech Integration Example

This example demonstrates TextToSpeech integration with EmailDataLoader and EmailClassifier,
showing practical audio capabilities for email management without requiring transformers.
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
from text_to_speech import TextToSpeech, TTSEngine, convert_email_summary_to_audio

def demonstrate_email_audio_integration():
    """Demonstrate TextToSpeech integration with email processing."""
    print("\n" + "="*80)
    print("MAILMATE EMAIL AUDIO INTEGRATION")
    print("="*80)
    
    # Initialize components
    print("Initializing MailMate components...")
    
    try:
        loader = EmailDataLoader()
        classifier = EmailClassifier()
        tts = TextToSpeech(engine=TTSEngine.AUTO)
        
        print("✓ All components initialized successfully")
        
        # Create output directories
        audio_dir = Path("audio_outputs/email_integration")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Generate emails and train classifier
        print(f"\nPhase 1: Email Generation and Classification")
        print("-" * 50)
        
        emails = loader.generate_synthetic_emails(count=150)
        print(f"Generated {len(emails)} synthetic emails")
        
        # Train classifier
        print("Training email classifier...")
        
        # Extract texts and labels from emails
        texts = [email.body for email in emails]
        labels = [email.category for email in emails]
        
        train_result = classifier.train(texts, labels)
        print(f"✓ Classifier trained with {train_result['test_accuracy']:.1%} accuracy")
        
        # Phase 2: Process sample emails with audio
        print(f"\nPhase 2: Email Processing with Audio Generation")
        print("-" * 50)
        
        # Sample emails for processing
        sample_emails = [
            {
                "subject": "Meeting Reminder - Project Kickoff",
                "content": "Don't forget about our project kickoff meeting tomorrow at 10 AM in Conference Room B. We'll be discussing project timelines, team responsibilities, and deliverables. Please bring your laptops and any relevant documents.",
                "sender": "project-manager@company.com"
            },
            {
                "subject": "IT Security Alert - Action Required",
                "content": "We have detected unusual activity on your account. As a precautionary measure, please update your password immediately and enable two-factor authentication. Contact IT support if you need assistance with this process.",
                "sender": "security@company.com"
            },
            {
                "subject": "Monthly Sales Report Available",
                "content": "The monthly sales report for October is now available on the company portal. This month shows a 12% increase in revenue compared to September. Please review the report and submit your analysis by Friday.",
                "sender": "sales-director@company.com"
            },
            {
                "subject": "Welcome New Team Member",
                "content": "Please join me in welcoming Sarah Johnson to our development team. Sarah brings 5 years of experience in software development and will be working on our mobile application project. Her first day is Monday.",
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
                
                # Step 2: Create concise summary for audio
                # Since we don't have the summarizer, create a simple summary
                words = email["content"].split()
                if len(words) > 30:
                    summary_text = " ".join(words[:25]) + "..."
                else:
                    summary_text = email["content"]
                
                print(f"  Summary: {summary_text[:80]}...")
                
                # Step 3: Convert to audio
                audio_result = convert_email_summary_to_audio(
                    summary_text=summary_text,
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
                    "summary_text": summary_text,
                    "audio": audio_result
                })
                
            except Exception as e:
                print(f"  ✗ Processing failed: {e}")
        
        # Phase 3: Category-based notifications
        print(f"\nPhase 3: Category-Based Audio Notifications")
        print("-" * 50)
        
        # Count emails by category
        category_counts = {}
        for result in processing_results:
            if result["audio"].success:
                category = result["classification"].predicted_category
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create notification texts
        notification_texts = []
        for category, count in category_counts.items():
            notification = f"You have {count} new {category} email{'s' if count > 1 else ''} waiting for your attention."
            notification_texts.append(notification)
        
        if notification_texts:
            print(f"Generating {len(notification_texts)} category notifications...")
            
            batch_results = tts.convert_batch(
                texts=notification_texts,
                output_directory=audio_dir / "notifications",
                filename_prefix="category_alert"
            )
            
            successful_notifications = sum(1 for r in batch_results if r.success)
            print(f"✓ Generated {successful_notifications}/{len(batch_results)} notification audio files")
            
            for i, result in enumerate(batch_results, 1):
                if result.success:
                    print(f"  ✓ Notification {i}: {result.processing_time:.2f}s")
                else:
                    print(f"  ✗ Notification {i}: {result.error_message}")
        
        # Phase 4: Urgent email alerts
        print(f"\nPhase 4: Urgent Email Audio Alerts")
        print("-" * 50)
        
        # Check for urgent emails (based on keywords)
        urgent_keywords = ["urgent", "emergency", "immediate", "asap", "critical", "alert"]
        urgent_emails = []
        
        for result in processing_results:
            email_text = (result["email"]["subject"] + " " + result["email"]["content"]).lower()
            if any(keyword in email_text for keyword in urgent_keywords):
                urgent_emails.append(result)
        
        if urgent_emails:
            print(f"Found {len(urgent_emails)} urgent emails")
            
            for i, urgent_email in enumerate(urgent_emails, 1):
                urgent_text = f"Urgent email alert: {urgent_email['email']['subject']}. This email requires immediate attention from {urgent_email['email']['sender']}."
                
                urgent_audio = tts.convert_text_to_speech(
                    text=urgent_text,
                    output_path=audio_dir / f"urgent_alert_{i}.mp3"
                )
                
                if urgent_audio.success:
                    print(f"  ✓ Urgent alert {i}: {urgent_audio.processing_time:.2f}s")
                else:
                    print(f"  ✗ Urgent alert {i}: {urgent_audio.error_message}")
        else:
            print("No urgent emails detected")
        
        # Phase 5: Daily summary
        print(f"\nPhase 5: Daily Email Summary Audio")
        print("-" * 50)
        
        total_processed = len(processing_results)
        successful_audio = sum(1 for r in processing_results if r["audio"].success)
        categories_list = list(category_counts.keys())
        
        daily_summary = f"""
        Daily email summary: {total_processed} emails were processed today. 
        {successful_audio} audio summaries were generated successfully. 
        Email categories include {', '.join(categories_list)}. 
        {'Urgent emails detected and flagged for immediate attention.' if urgent_emails else 'No urgent emails detected.'}
        All systems are operating normally.
        """
        
        daily_audio = tts.convert_text_to_speech(
            text=daily_summary.strip(),
            output_path=audio_dir / "daily_summary.mp3"
        )
        
        if daily_audio.success:
            print(f"✓ Daily summary audio generated: {daily_audio.processing_time:.2f}s")
        else:
            print(f"✗ Daily summary audio failed: {daily_audio.error_message}")
        
        # Final statistics
        print(f"\n" + "="*80)
        print("EMAIL AUDIO INTEGRATION SUMMARY")
        print("="*80)
        print(f"📧 Emails processed: {total_processed}")
        print(f"🎯 Classification accuracy: {train_result['test_accuracy']:.1%}")
        print(f"🔊 Audio files created: {successful_audio}")
        print(f"📂 Categories found: {len(category_counts)}")
        print(f"🚨 Urgent emails: {len(urgent_emails)}")
        print(f"📁 Audio output directory: {audio_dir}")
        
        # Show TTS engine info
        engine_info = tts.get_engine_info()
        print(f"\n🎙️ TTS Engine: {engine_info['current_engine']}")
        print(f"🌐 Language: {engine_info['language']}")
        print(f"⚙️ Available engines: {', '.join(engine_info['available_engines'])}")
        
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

def demonstrate_tts_voice_features():
    """Demonstrate TextToSpeech voice and language features."""
    print("\n" + "="*80)
    print("TEXT-TO-SPEECH VOICE AND LANGUAGE FEATURES")
    print("="*80)
    
    try:
        # Test different languages
        languages_to_test = ['en', 'es', 'fr', 'de']
        test_text = "This is a test of the text-to-speech system."
        
        audio_dir = Path("audio_outputs/language_tests")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        print("Testing multiple languages:")
        
        for lang in languages_to_test:
            try:
                tts = TextToSpeech(language=lang)
                
                # Translate test text for different languages (simple examples)
                if lang == 'es':
                    text = "Esta es una prueba del sistema de texto a voz."
                elif lang == 'fr':
                    text = "Ceci est un test du système de synthèse vocale."
                elif lang == 'de':
                    text = "Dies ist ein Test des Text-zu-Sprache-Systems."
                else:
                    text = test_text
                
                result = tts.convert_text_to_speech(
                    text=text,
                    output_path=audio_dir / f"test_{lang}.mp3"
                )
                
                if result.success:
                    print(f"  ✓ {lang}: {result.processing_time:.2f}s ({result.file_size_bytes:,} bytes)")
                else:
                    print(f"  ✗ {lang}: {result.error_message}")
                    
            except Exception as e:
                print(f"  ✗ {lang}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Voice features demonstration failed: {e}")
        return False

def main():
    """Run complete email audio integration demonstration."""
    print("MailMate AI Email Management System")
    print("Email Audio Integration Demonstration")
    print("=" * 80)
    
    try:
        # Run main integration
        integration_success = demonstrate_email_audio_integration()
        
        # Run voice features demo
        voice_success = demonstrate_tts_voice_features()
        
        if integration_success and voice_success:
            print(f"\n🎉 All demonstrations completed successfully!")
            print(f"📁 Check the 'audio_outputs' directory for generated audio files.")
            print(f"🎧 You can play these files to hear the text-to-speech results.")
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