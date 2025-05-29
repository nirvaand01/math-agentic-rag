import os
import dspy
from dspy.teleprompt import OpenAI
from dotenv import load_dotenv

load_dotenv()

def setup_dspy():
    """Configure DSPy with OpenAI settings."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
        
    # Configure DSPy to use GPT-4
    dspy.settings.configure(
        lm=OpenAI(
            model="gpt-4",
            api_key=api_key,
            max_tokens=1000
        )
    ) 