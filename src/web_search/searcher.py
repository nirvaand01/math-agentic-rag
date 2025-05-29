from typing import List, Dict
from tavily import TavilyClient
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os

load_dotenv()

class MathWebSearcher:
    def __init__(self):
        """Initialize the web searcher with Tavily API."""
        # Try both possible environment variable names
        api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_AI_KEY")
        if not api_key:
            raise ValueError("Neither TAVILY_API_KEY nor TAVILY_AI_KEY environment variable found")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search for math-related content using Tavily API.
        Returns cleaned and relevant content.
        """
        # Enhance query for math-specific results
        enhanced_query = f"mathematics {query} solution steps explanation"
        
        search_results = self.client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=max_results
        )
        
        processed_results = []
        for result in search_results['results']:
            # Extract and clean content
            content = self._clean_content(result.get('content', ''))
            if content:
                processed_results.append({
                    'content': content,
                    'url': result.get('url', ''),
                    'title': result.get('title', '')
                })
        
        return processed_results

    def _clean_content(self, content: str) -> str:
        """Clean and format the extracted content."""
        # Remove extra whitespace and normalize text
        content = ' '.join(content.split())
        
        # Basic cleaning
        content = content.replace('\n', ' ').replace('\t', ' ')
        content = ' '.join(line.strip() for line in content.split('\n') if line.strip())
        
        return content

    def get_context(self, query: str) -> str:
        """Get combined context from web search results."""
        results = self.search(query)
        if not results:
            return ""
        
        # Combine relevant content
        combined_context = "\n\n".join(
            f"Source: {result['title']}\n{result['content']}"
            for result in results
        )
        
        return combined_context 