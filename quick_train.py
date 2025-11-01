"""
Quick training script for emotion detection models - OPTIMIZED VERSION
"""
import sys
import traceback
from main import train_model, predict_emotion

def main():
    print("🚀 Quick Training Script for Emotion Detection (OPTIMIZED)")
    print("=" * 60)
    
    try:
        print("Choose training mode:")
        print("1. FAST LSTM (recommended for quick testing) - ~2-3 minutes")
        print("2. Optimized Transformer (better accuracy) - ~5-8 minutes")
        print("3. Both models for comparison")
        
        choice = input("\nEnter your choice (1/2/3) or press Enter for LSTM: ").strip()
        if not choice:
            choice = "1"
            
        if choice == "1":
            # Train LSTM model (fastest option)
            print("\n🏃‍♂️ Training Fast LSTM model...")
            print("This should complete in 2-3 minutes...")
            
            model, tokenizer = train_model(
                model_type='lstm',
                max_epochs=5,  # LSTM trains faster, can afford more epochs
                batch_size=64,  # Larger batch size for LSTM
                max_samples=4000  # Use subset for quick training
            )
            
            model_name = "LSTM"
            
        elif choice == "2":
            # Train optimized transformer model
            print("\n🤖 Training Optimized DistilBERT model...")
            print("This should complete in 5-8 minutes...")
            
            model, tokenizer = train_model(
                model_type='transformer',
                max_epochs=3,  # Reduced from original
                batch_size=32,  # Increased from 16
                max_samples=4000  # Use subset for quick training
            )
            
            model_name = "Optimized Transformer"
            
        elif choice == "3":
            # Train both models for comparison
            print("\n🏃‍♂️ Training Fast LSTM model first...")
            
            model_lstm, tokenizer_lstm = train_model(
                model_type='lstm',
                max_epochs=5,
                batch_size=64,
                max_samples=3000
            )
            
            print("\n🤖 Training Optimized Transformer model...")
            
            model_transformer, tokenizer_transformer = train_model(
                model_type='transformer',
                max_epochs=3,
                batch_size=32,
                max_samples=3000
            )
            
            # Use transformer for final testing (usually more accurate)
            model, tokenizer = model_transformer, tokenizer_transformer
            model_name = "Both (using Transformer for testing)"
            
        else:
            print("Invalid choice. Using LSTM model...")
            model, tokenizer = train_model(
                model_type='lstm',
                max_epochs=5,
                batch_size=64,
                max_samples=4000
            )
            model_name = "LSTM"
        
        print(f"\n✅ Training completed for {model_name}!")
        print("=" * 60)
        
        # Test the trained model
        print("🧪 Testing the trained model...")
        
        # More diverse test cases
        test_texts = [
            "I am so excited about this new project!",  # joy
            "This makes me really angry and frustrated.",  # anger
            "I'm feeling quite sad and lonely today.",  # sadness
            "That's absolutely terrifying and scary!",  # fear
            "I love spending time with my family so much.",  # love
            "What a surprising and unexpected turn of events!",  # surprise
            "The weather is nice today.",  # neutral-ish
            "I hate when things go wrong like this.",  # anger
            "I'm so grateful for all the support.",  # love/joy
            "This situation is making me very worried."  # fear
        ]
        
        print(f"\nPredictions using {model_name}:")
        print("-" * 50)
        
        correct_predictions = 0
        total_predictions = len(test_texts)
        
        for i, text in enumerate(test_texts, 1):
            try:
                emotion = predict_emotion(text, model, tokenizer)
                print(f"{i:2d}. Text: '{text}'")
                print(f"    Emotion: {emotion.upper()}")
                print("-" * 50)
            except Exception as e:
                print(f"    Error predicting for text {i}: {e}")
        
        # If both models were trained, compare them
        if choice == "3":
            print("\n🔍 Comparing LSTM vs Transformer predictions:")
            print("-" * 60)
            
            sample_texts = test_texts[:5]  # Test on first 5 for comparison
            
            for i, text in enumerate(sample_texts, 1):
                try:
                    emotion_lstm = predict_emotion(text, model_lstm, tokenizer_lstm)
                    emotion_transformer = predict_emotion(text, model_transformer, tokenizer_transformer)
                    
                    print(f"{i}. '{text}'")
                    print(f"   LSTM:        {emotion_lstm.upper()}")
                    print(f"   Transformer: {emotion_transformer.upper()}")
                    if emotion_lstm != emotion_transformer:
                        print("   → Different predictions!")
                    print("-" * 60)
                    
                except Exception as e:
                    print(f"   Error comparing predictions for text {i}: {e}")
        
        print("\n🎉 Quick training and testing completed successfully!")
        print("\nModel Performance Summary:")
        print(f"• Model type: {model_name}")
        print(f"• Training time: Significantly reduced with optimizations")
        print(f"• Memory usage: Optimized for CPU training")
        
        print("\n📁 Saved Files:")
        if choice == "1":
            print("• emotion_lstm_model.ckpt")
        elif choice == "2":
            print("• emotion_transformer_model.ckpt")
        elif choice == "3":
            print("• emotion_lstm_model.ckpt")
            print("• emotion_transformer_model.ckpt")
            
        print("\n🚀 Next steps:")
        print("1. Run 'streamlit run streamlit_app.py' to start the web interface")
        print("2. Use the saved model files for inference in your applications")
        print("3. For production: train on full dataset without max_samples limit")
        
        # Performance tips
        print("\n💡 Performance Tips:")
        print("• LSTM model: ~5-10x faster training, good for quick experiments")
        print("• Transformer model: Better accuracy, optimized for CPU training")
        print("• For production: remove max_samples parameter for full dataset")
        print("• Adjust batch_size based on your CPU cores and RAM")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Tip: You can resume training by running the script again")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nFull error trace:")
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Make sure all packages are installed: pip install -r requirements.txt")
        print("2. Check your internet connection (needed to download models)")
        print("3. Ensure you have enough disk space (models are ~250MB)")
        print("4. Try reducing batch_size if you get memory errors:")
        print("   - For LSTM: try batch_size=32")
        print("   - For Transformer: try batch_size=16")
        print("5. If still failing, try max_samples=1000 for even faster training")
        
        # Additional CPU-specific troubleshooting
        print("\n🖥️ CPU-specific troubleshooting:")
        print("• Close other applications to free up CPU and RAM")
        print("• The optimizations should work on any modern CPU")
        print("• LSTM model is much faster if you're still having speed issues")

if __name__ == "__main__":
    main()