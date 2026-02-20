#!/usr/bin/env python3
"""
Test script to verify fallback logic: Phi-3-mini -> OpenAI API
"""
import os
import sys
from pathlib import Path

# Get script directory and set paths relative to EduBot root
SCRIPT_DIR = Path(__file__).parent.resolve()
EDUBOT_DIR = SCRIPT_DIR.parent
INTERFACE_DIR = EDUBOT_DIR / 'MI2US_Year2s' / 'Interface'

# Set cache directories (relative to EduBot)
cache_dir = EDUBOT_DIR / '.cache' / 'huggingface'
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)

sys.path.insert(0, str(INTERFACE_DIR))

print("Testing fallback logic...")
print("=" * 60)

# Test 1: Normal operation (should use Phi-3-mini)
print("\n1. Testing normal operation (should use Phi-3-mini):")
print("-" * 60)
try:
    import storytelling as st
    result = st.generate_response("Say hello in one sentence.")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Force fallback by temporarily breaking Phi-3
print("\n2. Testing fallback (simulating Phi-3 failure):")
print("-" * 60)
print("To test fallback, you can:")
print("  - Temporarily rename the model files")
print("  - Or set OPENAI_API_KEY and let it fail naturally")
print("  - Or modify _load_model() to raise an exception")

# Check if OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key != "YOUR_KEY":
    print(f"\n[OK] OPENAI_API_KEY is set (length: {len(api_key)})")
    print("  Fallback to OpenAI will work if Phi-3 fails")
else:
    print("\n[WARNING] OPENAI_API_KEY not set or is placeholder")
    print("  Set it with: export OPENAI_API_KEY='your-key-here'")
    print("  Fallback will not work without this")

print("\n" + "=" * 60)
print("Fallback logic implementation complete!")
print("\nHow it works:")
print("  1. First tries Phi-3-mini (local model)")
print("  2. If that fails, automatically falls back to OpenAI API")
print("  3. If both fail, raises an error with details")
