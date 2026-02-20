#!/usr/bin/env python3
"""
Direct test of Phi-Ed story generation (no WebSocket)
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

print("Loading storytelling module...")
import storytelling

print("\nTesting Phi-Ed story generation\n")

# Test story prompt
topic = "a robot learning to help children"
age_level = "Preschoolers"  # Valid: Preschoolers, Early elementary, Late elementary, Preteens
story_length = 300

print(f"Topic: {topic}")
print(f"Age: {age_level}")
print(f"Length: {story_length} words")
print("\n" + "="*60)
print("GENERATING STORY...")
print("="*60 + "\n")

try:
    # Call the story generation function (dSgDaG = Design Story given Demographics and Genre)
    story = storytelling.dSgDaG(
        topic=topic,
        age_group=age_level,
        word_count=story_length,
        temperature=0.8,
        max_tokens=1000
    )
    
    print("[SUCCESS] Story generated successfully!\n")
    print("="*60)
    print("STORY:")
    print("="*60)
    print(story)
    print("="*60)
    print(f"\nStory length: {len(story.split())} words")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
