from typing import List, Dict, Optional
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

class WebSearchVerifier:
    def __init__(self):
        """Initialize the web search verifier with API clients."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")

    def search_tavily(self, query: str) -> List[Dict]:
        """Search using Tavily API."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.tavily_api_key
        }
        
        data = {
            "query": f"math problem solution: {query}",
            "search_depth": "advanced",
            "max_results": 3
        }
        
        response = requests.post(
            "https://api.tavily.com/search",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def search_serper(self, query: str) -> List[Dict]:
        """Search using Serper API."""
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "q": f"math problem solution step by step: {query}",
            "num": 3
        }
        
        response = requests.post(
            "https://google.serper.dev/search",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get("organic", [])
        return []

    def verify_sources(self, query: str, sources: List[Dict]) -> Dict:
        """Verify and compare multiple search results."""
        if not sources:
            return {"verified": False, "reason": "No sources found"}
            
        # Combine source content for comparison
        source_texts = []
        for source in sources:
            if "snippet" in source:  # Tavily format
                source_texts.append(source["snippet"])
            elif "snippet" in source:  # Serper format
                source_texts.append(source["snippet"])
        
        if len(source_texts) < 2:
            return {"verified": False, "reason": "Insufficient sources for verification"}
            
        # Compare sources using GPT-4
        prompt = f"""Compare these search results for the math problem and determine if they are consistent.
Answer ONLY with 'consistent' or 'inconsistent' followed by a brief explanation.

Question: {query}

Sources:
{chr(10).join(f'Source {i+1}: {text}' for i, text in enumerate(source_texts))}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a math expert verifying the consistency of multiple sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.lower()
        return {
            "verified": result.startswith("consistent"),
            "reason": result.split(maxsplit=1)[1] if len(result.split(maxsplit=1)) > 1 else ""
        }

    def search_and_verify(self, query: str) -> Dict:
        """Perform web search with multi-source verification."""
        # Get results from both search engines
        tavily_results = self.search_tavily(query)
        serper_results = self.search_serper(query)
        
        # Combine and verify results
        all_results = tavily_results + serper_results
        verification = self.verify_sources(query, all_results)
        
        return {
            "query": query,
            "sources": all_results,
            "verification": verification,
            "can_proceed": verification["verified"]
        } 