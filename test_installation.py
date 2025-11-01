"""
Test script to verify all components are working
"""
import sys
import traceback

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
        return False
    
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch Lightning not found: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers not found: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets not found: {e}")
        return False
    
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit not found: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if the emotion dataset can be loaded"""
    print("\nTesting data loading...")
    
    try:
        from datasets import load_dataset
        print("Attempting to load emotion dataset...")
        dataset = load_dataset("emotion")
        print(f"✅ Dataset loaded successfully")
        print(f"   - Train samples: {len(dataset['train'])}")
        print(f"   - Validation samples: {len(dataset['validation'])}")
        print(f"   - Test samples: {len(dataset['test'])}")
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        # Try importing the main module
        try:
            from main import LSTMEmotionClassifier, TransformerEmotionClassifier
            print("✅ Successfully imported model classes from main.py")
        except ImportError as e:
            print(f"❌ Could not import from main.py: {e}")
            print("   Make sure main.py exists and contains the model classes")
            return False
        
        # Test LSTM
        lstm_model = LSTMEmotionClassifier(vocab_size=1000, num_classes=6)
        print("✅ LSTM model created successfully")
        
        # Test Transformer
        transformer_model = TransformerEmotionClassifier(num_classes=6)
        print("✅ Transformer model created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    try:
        print("🧪 Testing Emotion Detection Setup")
        print("=" * 40)
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print("=" * 40)
        
        all_tests_passed = True
        
        # Run tests with individual error handling
        try:
            all_tests_passed &= test_imports()
        except Exception as e:
            print(f"❌ Import test failed with exception: {e}")
            traceback.print_exc()
            all_tests_passed = False
        
        try:
            all_tests_passed &= test_data_loading()
        except Exception as e:
            print(f"❌ Data loading test failed with exception: {e}")
            traceback.print_exc()
            all_tests_passed = False
        
        try:
            all_tests_passed &= test_model_creation()
        except Exception as e:
            print(f"❌ Model creation test failed with exception: {e}")
            traceback.print_exc()
            all_tests_passed = False
        
        print("\n" + "=" * 40)
        if all_tests_passed:
            print("🎉 All tests passed! Your setup is ready to go!")
            print("\nNext steps:")
            print("1. Run 'python quick_train.py' to train models")
            print("2. Run 'streamlit run streamlit_app.py' to start the web app")
        else:
            print("❌ Some tests failed. Please check the error messages above.")
            print("Make sure you've installed all requirements with:")
            print("pip install -r requirements.txt")
    
    except Exception as e:
        print(f"❌ Script failed with unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Script execution completed ---")