import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Add src to path to import constants if needed, but we'll define a simple prompt here
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

SYSTEM_PROMPT = "Du bist ein KI-Assistent. Antworte immer im JSON-Format."
USER_PROMPT = "Analysiere: Solar Modul 450W Trina Vertex S+. Klassifiziere es."

def debug_gpt5(model_name="gpt-5-mini"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY missing")
        return

    client = OpenAI(api_key=api_key, timeout=1200.0)  # 20min timeout for GPT-5 reasoning
    
    print(f"🧪 Testing {model_name} with SINGLE request...")
    print("   - Temperature: Removed")
    print("   - Response Format: JSON")
    
    try:
        # Attempt 1: Standard JSON Request (no temp)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            response_format={"type": "json_object"}
            # temperature is OMITTED
        )
        
        print("\n✅ SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        if hasattr(response, 'usage'):
             print(f"Usage: {response.usage}")
             
    except Exception as e:
        print(f"\n❌ FAILED with JSON Mode: {e}")
        
        # Attempt 2: Retry WITHOUT Response Format (Plain text)
        print("\n🔄 Retrying WITHOUT 'response_format' (Plain Text Mode)...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": SYSTEM_PROMPT + "\n" + USER_PROMPT} # Move system to user for o1-compat
                ]
                # No system role (o1-preview specific handling if needed)
                # No response_format
                # No temperature
            )
            print("\n✅ SUCCESS (Plain Text)!")
            print(f"Response: {response.choices[0].message.content}")
            
        except Exception as e2:
             print(f"\n❌ FAILED Plain Text: {e2}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-mini"
    debug_gpt5(model)
