import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from src.feedback.feedback_manager import FeedbackManager, FeedbackType
from src.feedback.feedback_loop import FeedbackManager as FeedbackLoopManager
from datetime import datetime
import openai

# Initialize DSPy with OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

openai.api_key = api_key
lm = dspy.LM("openai/gpt-4", api_key=api_key)
dspy.settings.configure(lm=lm)

def test_feedback_collection():
    """Test the basic feedback collection functionality."""
    # Initialize feedback manager with a test file
    feedback_manager = FeedbackManager("test_feedback_history.json")
    
    # Sample math problem and solution
    test_problem = "Solve the quadratic equation: x² + 5x + 6 = 0"
    test_solution = {
        "steps": [
            "Using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a",
            "Here, a = 1, b = 5, c = 6",
            "x = (-5 ± √(25 - 24)) / 2",
            "x = (-5 ± √1) / 2",
            "x = (-5 ± 1) / 2"
        ],
        "answer": "x = -3 or x = -2",
        "explanation": "The solution can be verified by plugging these values back into the original equation."
    }
    
    print("\n=== Testing Feedback Collection ===")
    print(f"\nProblem: {test_problem}")
    print("\nSolution:")
    for step in test_solution["steps"]:
        print(f"- {step}")
    print(f"Answer: {test_solution['answer']}")
    
    # Collect feedback
    feedback = feedback_manager.collect_feedback(test_problem, test_solution)
    
    if feedback:
        print("\nFeedback collected successfully!")
        print("\nFeedback Statistics:")
        stats = feedback_manager.get_feedback_stats()
        print(f"Total feedback entries: {stats['total_feedback']}")
        print("\nAverage ratings:")
        for criteria, rating in stats['average_ratings'].items():
            print(f"- {criteria}: {rating:.2f}")

def test_feedback_refinement():
    """Test the feedback-based solution refinement."""
    feedback_loop = FeedbackLoopManager()
    
    # Sample problem and initial solution
    question = "Find the derivative of f(x) = x³ + 2x² - 4x + 1"
    solution = {
        "steps": [
            "Using power rule for each term:",
            "For x³: derivative is 3x²",
            "For 2x²: derivative is 4x",
            "For -4x: derivative is -4",
            "For 1: derivative is 0"
        ],
        "answer": "f'(x) = 3x² + 4x - 4"
    }
    
    print("\n=== Testing Solution Refinement ===")
    print(f"\nOriginal Problem: {question}")
    print("Original Solution:")
    for step in solution["steps"]:
        print(f"- {step}")
    print(f"Answer: {solution['answer']}")
    
    # Collect feedback and refine
    feedback = feedback_loop.collect_feedback()
    if feedback:
        feedback.original_solution = solution  # Set the original solution
        refined_solution = feedback_loop.refine_solution(question, solution, feedback)
        if refined_solution:
            print("\nRefined Solution:")
            print(refined_solution)

if __name__ == "__main__":
    print("Starting feedback system tests...")
    test_feedback_collection()
    test_feedback_refinement()
    print("\nTests completed!") 