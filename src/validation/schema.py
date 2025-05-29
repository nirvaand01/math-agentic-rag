from pydantic import BaseModel, validator
import re

class MathQuery(BaseModel):
    """Schema for validating math-related queries."""
    query: str

    @validator('query')
    def validate_math_content(cls, v):
        # Math symbols and keywords to check for
        math_patterns = [
            r'[\+\-\*/\^\(\)\[\]\{\}=<>≤≥≠∫∑∏√]',  # Math symbols
            r'\b(solve|prove|calculate|find|integrate|differentiate|evaluate|simplify)\b',  # Math verbs
            r'\b(equation|function|derivative|integral|limit|series|matrix|vector|polynomial)\b',  # Math nouns
            r'\b(sin|cos|tan|log|ln|exp)\b',  # Math functions
            r'\b(algebra|calculus|geometry|trigonometry|statistics|probability)\b',  # Math subjects
            r'\b\d+\b',  # Numbers
            r'\b[xyz]\b',  # Common variables
            r'\b(pi|infinity|inf)\b'  # Math constants
        ]
        
        # Check if query contains at least one math pattern
        is_math_related = any(re.search(pattern, v.lower()) for pattern in math_patterns)
        
        # List of non-math keywords to filter out
        non_math_keywords = [
            r'\b(poem|story|essay|write|compose|create|generate)\b',
            r'\b(song|music|lyrics|dance|paint|draw)\b',
            r'\b(recipe|cook|bake|food|drink)\b',
            r'\b(joke|funny|humor|comedy)\b'
        ]
        
        # Check if query contains any non-math keywords
        has_non_math = any(re.search(pattern, v.lower()) for pattern in non_math_keywords)
        
        if not is_math_related:
            raise ValueError("Query must contain mathematical terms, symbols, or concepts")
        
        if has_non_math:
            raise ValueError("Query contains non-mathematical terms that are not allowed")
            
        # Check minimum and maximum length
        if len(v) < 5:
            raise ValueError("Query is too short")
        if len(v) > 500:
            raise ValueError("Query is too long")
            
        return v

    @validator('query')
    def validate_complexity(cls, v):
        """Validate that the query isn't too complex or too simple."""
        # Count mathematical symbols and terms
        math_symbols = len(re.findall(r'[\+\-\*/\^\(\)\[\]\{\}=<>≤≥≠∫∑∏√]', v))
        
        # Very complex expressions might indicate copy-pasted content
        if math_symbols > 50:
            raise ValueError("Query is too complex. Please break it down into smaller parts")
            
        return v 