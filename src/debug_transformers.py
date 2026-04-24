import traceback
import sys

print("Attempting to import transformers.pipeline...")
try:
    from transformers import pipeline
    print("Success: transformers.pipeline imported.")
    p = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    print("Success: Pipeline initialized.")
except ImportError as ie:
    print(f"FAILED with ImportError: {ie}")
    traceback.print_exc()
except Exception as e:
    print(f"FAILED with generic Exception: {type(e).__name__}: {e}")
    traceback.print_exc()
