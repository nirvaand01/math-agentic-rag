from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class MCPGenerator:
    def __init__(self):
        """Initialize the MCP generator with OpenAI."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.base_prompt = """You are an expert math professor who creates similar math problems.
Given a math problem, generate {num_variations} similar problems that test the same concepts but with:
1. Different numbers/variables
2. Different difficulty levels (easier and harder)
3. Different contexts or applications

Format each problem as a JSON object with:
- question: The problem statement
- concepts: List of mathematical concepts tested
- difficulty: "easy", "medium", or "hard"
- solution: Step-by-step solution
"""

    def generate_similar_problems(self, query: str, num_variations: int = 3) -> List[Dict]:
        """Generate similar math problems using GPT-4."""
        prompt = self.base_prompt.format(num_variations=num_variations)
        prompt += f"\n\nOriginal Problem: {query}\n\nGenerate {num_variations} similar problems in JSON format."
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please generate similar problems with complete solutions."}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            # Parse the JSON response
            variations = eval(response.choices[0].message.content)
            return variations.get("problems", [])
        except Exception as e:
            print(f"Error parsing MCP response: {e}")
            return []

    def enhance_search_context(self, query: str) -> str:
        """Generate search context by analyzing the problem."""
        prompt = """Analyze this math problem and provide:
1. The key mathematical concepts involved
2. Related topics that might be helpful
3. Common solution approaches
4. Similar types of problems

Format your response as a clear, concise paragraph."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content 