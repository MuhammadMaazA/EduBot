"""
Test the existing OpenAI storytelling code
"""
import os
import sys
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: No API key found in .env file")
    sys.exit(1)

print("[OK] API key loaded from .env")
print(f"[OK] Key starts with: {api_key[:20]}...")

# Update the storytelling.py file with the API key
storytelling_path = "MI2US_Year2s/Interface/storytelling.py"

print(f"\n[UPDATING] {storytelling_path} with API key...")

with open(storytelling_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the placeholder API keys
content = content.replace('OpenAI.api_key ="YOUR_KEY"', f'OpenAI.api_key ="{api_key}"')
content = content.replace('os.environ["OPENAI_API_KEY"] = "YOUR_KEY"', f'os.environ["OPENAI_API_KEY"] = "{api_key}"')

with open(storytelling_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] API key updated in storytelling.py")

# Now try to import and test
print("\n[TESTING] Story generation...")
sys.path.insert(0, "MI2US_Year2s/Interface")

try:
    import storytelling as ai
    
    print("[OK] storytelling module imported")
    
    # Test simple story generation
    print("\n[GENERATING] Short story for 5-year-olds about robots...")
    story = ai.dSgDaG(
        topic="A friendly robot helper",
        age_group="Preschoolers",
        word_count=100
    )
    
    print("\n" + "="*60)
    print("GENERATED STORY:")
    print("="*60)
    print(story)
    print("="*60)
    
    print("\n[SUCCESS] OpenAI integration is working!")
    print("\nNext steps:")
    print("   1. This confirms the code works with OpenAI")
    print("   2. Now you can proceed to replace OpenAI with Phi-Ed")
    print("   3. Keep the prompt engineering - it's excellent!")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure you have installed: pip install openai python-dotenv")
