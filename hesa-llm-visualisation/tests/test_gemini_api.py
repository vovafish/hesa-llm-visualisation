"""
Test script for Gemini API functionality
This script generates a short tale and a table to verify that the API is working properly.
"""

from google import genai
import os
import sys
from dotenv import load_dotenv

def test_gemini_api():
    # Load API key from data.env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hesa-llm-visualisation", "data.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        print(f"Warning: data.env file not found at {env_path}")
        api_key = "AIzaSyA8DVRyYopbD9FIuBEgRPUDkAvBnhR5ZO0"  # Fallback to hardcoded key
    
    print(f"Using API key from: {'data.env file' if os.path.exists(env_path) else 'fallback hardcoded value'}")
    
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)
    
    print("\n🔄 Testing Gemini API connection...")
    
    # Generate a short tale
    tale_prompt = "Create a very short tale (no more than 100 words) about a friendly dragon who likes to bake cookies."
    
    try:
        tale_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=tale_prompt
        )
        
        print("\n✅ API Connection Successful!\n")
        print("🐉 SHORT TALE:\n")
        print(tale_response.text)
        
        # Generate a simple table
        table_prompt = "Create a small table showing 3 types of cookies, their main ingredients, and baking temperature in a simple text format."
        
        table_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=table_prompt
        )
        
        print("\n📊 TABLE:\n")
        print(table_response.text)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - Network connectivity problems")
        print("  - API quota exceeded")
        print("  - Model name incorrect (check if 'gemini-2.0-flash' is available)")

if __name__ == "__main__":
    test_gemini_api() 