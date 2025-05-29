from dotenv import load_dotenv
import os

print("Loading environment variables...")
load_dotenv(verbose=True)

print("\nChecking environment variables:")
print("TAVILY_AI_KEY present:", "Yes" if os.getenv("TAVILY_AI_KEY") else "No")
print("OPENAI_API_KEY present:", "Yes" if os.getenv("OPENAI_API_KEY") else "No")

# Print all env var names (not values) for debugging
print("\nAll environment variable names:")
for key in os.environ:
    if "API" in key or "AI" in key:  # Updated to catch AI keys too
        print(f"Found: {key}") 