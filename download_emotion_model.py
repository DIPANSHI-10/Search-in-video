#!/usr/bin/env python3
"""
Pre-download the emotion classification model.
Run this once to download the model to your local cache.
"""

import os

def download_emotion_model():
    print("=" * 60)
    print("Setting Up Zero-Shot Emotion Classification")
    print("=" * 60)
    print("\nThis may take a minute on first run...")
    print("(The model will be cached locally after this)\n")
    
    try:
        from transformers import pipeline
        
        print("📥 Loading zero-shot classification model...")
        print("   (This downloads ~350MB the first time)")
        
        # Use zero-shot classification which works without needing specific trained emotion model
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        
        print("✅ Emotion classification model downloaded and ready!")
        
        # Test it
        test_text = "I am very happy and excited!"
        candidate_emotions = ["happy", "sad", "angry", "scared", "neutral"]
        result = classifier(test_text, candidate_emotions)
        print(f"\n✨ Test Result:")
        print(f"   Input: {test_text}")
        print(f"   Emotions: {result}")
        print("\n" + "=" * 60)
        print("✅ Everything is set up! You can now run the app:")
        print("   streamlit run src/main.py")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: transformers library not installed")
        print(f"   Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_emotion_model()
    exit(0 if success else 1)
