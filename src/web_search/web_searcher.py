import os
from typing import List, Dict, Any
from tavily import TavilyClient
import re

class WebSearcher:
    """Class to search for math-related information on the web using Tavily API."""
    
    def __init__(self):
        """Initialize the web searcher."""
        self.api_key = os.getenv("TAVILY_AI_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key not found. Please set TAVILY_AI_KEY environment variable.")
        self.client = TavilyClient(api_key=self.api_key)
    
    def extract_math_concepts(self, query: str) -> str:
        """Extract key math concepts from the query."""
        # Common math terms to look for
        math_terms = [
            r"trigonometric|sin|cos|tan|sec|cosec|cot",
            r"quadratic|equation|root|polynomial",
            r"matrix|determinant|inverse",
            r"derivative|integral|calculus",
            r"vector|scalar|dot product|cross product",
            r"probability|statistics|mean|median|mode",
            r"geometry|circle|triangle|square|polygon",
            r"logarithm|exponential|power",
            r"complex number|imaginary|real",
            r"series|sequence|progression"
        ]
        
        # Find all math terms in the query
        found_terms = set()
        for pattern in math_terms:
            matches = re.findall(pattern, query.lower())
            found_terms.update(matches)
        
        # Extract numbers and mathematical symbols
        math_symbols = re.findall(r'[\+\-\*/\^=<>≤≥≠∫∑∏√]+', query)
        found_terms.update(math_symbols)
        
        # If no terms found, take the first 50 characters
        if not found_terms:
            return query[:50]
        
        # Combine terms into a search query
        search_terms = " ".join(found_terms)
        return f"math problem {search_terms}"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for math-related information using Tavily API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, content, and URL
        """
        try:
            # Extract key math concepts for shorter query
            enhanced_query = self.extract_math_concepts(query)
            
            # Perform search
            response = self.client.search(
                query=enhanced_query,
                search_depth="advanced",
                max_results=max_results
            )
            
            # Format results
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                })
            
            return results
            
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
            return [] 