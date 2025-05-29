import dspy
from dspy.teleprompt import OpenAI
from typing import Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from .validation.schema import MathQuery
from .knowledge_base.vectorstore import MathKnowledgeBase
from .web_search.search import WebSearchVerifier
from .solution.formatter import SolutionFormatter
from .feedback.feedback_loop import FeedbackManager

load_dotenv()

# Configure DSPy
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

dspy.settings.configure(
    lm=OpenAI(
        api_key=api_key,
        model="gpt-4"
    )
)

class MathAgent:
    def __init__(self):
        """Initialize the Math Agent with all components."""
        self.knowledge_base = MathKnowledgeBase()
        self.web_search = WebSearchVerifier()
        self.formatter = SolutionFormatter()
        self.feedback_manager = FeedbackManager()
        
    def solve_problem(self, query: str) -> Optional[Dict]:
        """Solve a math problem with human feedback loop."""
        try:
            # Initial solution
            solution = self._get_initial_solution(query)
            if not solution:
                return None
                
            # Get human feedback
            print("\nInitial solution:")
            print(solution["solution"])
            
            # Collect feedback
            feedback = self.feedback_manager.collect_feedback()
            feedback.original_solution = solution
            
            # If rating is less than 4, refine the solution
            if feedback.rating < 4:
                print("\nRefining solution based on feedback...")
                refined_solution = self.feedback_manager.refine_solution(query, solution, feedback)
                print("\nRefined solution:")
                print(refined_solution["solution"])
                return refined_solution
            
            return solution
            
        except Exception as e:
            print(f"Error solving problem: {str(e)}")
            return None
            
    def _get_initial_solution(self, query: str) -> Optional[Dict]:
        """Get the initial solution using existing pipeline."""
        # Validate query
        math_query = MathQuery(query=query)
        
        # Try knowledge base first
        solution = self.knowledge_base.find_similar_problem(math_query.query)
        
        # If no good match in knowledge base, use web search
        if not solution or solution.get("confidence", 0) < 0.8:
            solution = self.web_search.search_and_verify(math_query.query)
            
        # Format the solution
        if solution:
            return self.formatter.format_solution(solution)
        return None

def main():
    """Main entry point for the math solver."""
    agent = MathAgent()
    print("\nMath Problem Solver initialized!")
    print("Type 'quit' to exit")
    
    while True:
        print("\nEnter your math problem:")
        query = input("> ")
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
            
        solution = agent.solve_problem(query)
        if not solution:
            print("\nSorry, I couldn't solve that problem. Please try rephrasing it.")

if __name__ == "__main__":
    main() 