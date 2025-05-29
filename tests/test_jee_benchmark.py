import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import dspy
from src.feedback.feedback_manager import FeedbackManager
from src.llm.solution_generator import SolutionGenerator
from src.knowledge_base.vector_store import KnowledgeBase
from src.web_search.web_searcher import WebSearcher
from src.validation.input_validator import MathQuery
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug prints
print("Environment variables check:")
print("TAVILY_AI_KEY present:", "Yes" if os.getenv("TAVILY_AI_KEY") else "No")
print("OPENAI_API_KEY present:", "Yes" if os.getenv("OPENAI_API_KEY") else "No")

def load_jee_dataset():
    """Load JEE benchmark dataset."""
    # Download and unzip the dataset
    os.system("wget -q https://github.com/dair-iitd/jeebench/raw/main/data.zip")  # Added -q for quiet download
    os.system("yes | unzip -o data.zip")  # Added yes and -o to automatically overwrite
    
    # Load the dataset
    with open("data/dataset.json", "r") as f:
        dataset = json.load(f)
    return dataset

def extract_answer_from_text(text: str) -> str:
    """Extract the final answer from the solution text."""
    text = text.lower()
    
    # Look for common answer patterns
    patterns = [
        r"therefore,?\s*the\s*answer\s*is\s*\(?([A-Da-d])\)?",
        r"the\s*answer\s*is\s*\(?([A-Da-d])\)?",
        r"final\s*answer\s*is\s*\(?([A-Da-d])\)?",
        r"answer:\s*\(?([A-Da-d])\)?",
        r"([A-Da-d])\s*is\s*the\s*correct\s*answer",
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    
    # If no pattern matches, fall back to the last line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # Extract any single letter A-D from the last line
        letters = re.findall(r'[A-Da-d]', last_line)
        if letters:
            return letters[-1].upper()
    
    return text

def clean_answer(answer: str, question_type: str) -> str:
    """Clean and normalize the answer string based on question type."""
    if not answer:
        return ""
        
    answer = str(answer).strip()
    
    if question_type in ["MCQ", "MCQ(multiple)"]:
        # First try to extract answer from text if it's a longer response
        if len(answer) > 5:  # If it's more than just a letter or two
            answer = extract_answer_from_text(answer)
        # Extract only letters and convert to uppercase
        cleaned = ''.join(c for c in answer if c.isalpha())
        return cleaned.upper()
    elif question_type in ["Integer", "Numeric"]:
        # Extract numbers, decimal points, and negative signs
        # First, find any number in the text (including decimals and negatives)
        import re
        numbers = re.findall(r'-?\d*\.?\d+', answer)
        if numbers:
            return numbers[0]  # Return the first number found
        return ""
    return answer

def evaluate_model(dataset, model):
    """Evaluate model on JEE benchmark dataset."""
    # Filter for math problems only
    math_problems = [q for q in dataset if q["subject"] == "math"]
    
    results = {
        "total": 0,
        "correct": 0,
        "type_wise": {
            "MCQ": {"total": 0, "correct": 0},
            "MCQ(multiple)": {"total": 0, "correct": 0},
            "Integer": {"total": 0, "correct": 0},
            "Numeric": {"total": 0, "correct": 0}
        }
    }
    
    total_questions = len(math_problems)
    print(f"\nStarting evaluation on {total_questions} JEE math questions...")
    
    print("\nProgress by type:")
    for qtype in ["MCQ", "MCQ(multiple)", "Integer", "Numeric"]:
        type_count = len([q for q in math_problems if q["type"] == qtype])
        print(f"{qtype}: {type_count} questions")
    
    print("\nProcessing questions...")
    for i, question in enumerate(tqdm(math_problems, desc="Math Questions")):
        # Print detailed progress every 5 questions
        if i % 5 == 0:
            print(f"\nQuestion {i+1}/{total_questions}")
            print(f"Type: {question['type']}")
            print("Current accuracy:", f"{(results['correct']/results['total']*100):.2f}%" if results['total'] > 0 else "N/A")
        
        # Prepare input
        query = MathQuery(
            question=question["question"],
            subject="math",
            type=question["type"]
        )
        
        # Generate solution
        solution = model.solve(query)
        
        # Clean and compare answers
        predicted_answer = clean_answer(solution.get("answer", ""), question["type"])
        correct_answer = clean_answer(question["gold"], question["type"])
        
        # Update statistics
        results["total"] += 1
        results["type_wise"][question["type"]]["total"] += 1
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            results["correct"] += 1
            results["type_wise"][question["type"]]["correct"] += 1
            
        # Print result for this question
        print(f"\nQuestion {i+1} Result:")
        print(f"Question type: {question['type']}")
        print(f"Raw predicted: {solution.get('answer', '')}")
        print(f"Cleaned predicted: {predicted_answer}")
        print(f"Raw correct: {question['gold']}")
        print(f"Cleaned correct: {correct_answer}")
        print(f"Status: {'✓' if is_correct else '✗'}")
        if not is_correct:
            print("Full solution:", solution.get("solution", ""))
    
    # Calculate accuracies
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    for qtype in results["type_wise"]:
        if results["type_wise"][qtype]["total"] > 0:
            results["type_wise"][qtype]["accuracy"] = results["type_wise"][qtype]["correct"] / results["type_wise"][qtype]["total"]
    
    return results

def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n=== JEE Math Questions Evaluation Results ===")
    print(f"\nOverall Accuracy: {results['accuracy']:.2%}")
    
    print("\nQuestion Type-wise Performance:")
    for qtype, stats in results["type_wise"].items():
        if stats["total"] > 0:
            print(f"{qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

def test_single_problem(dataset, model):
    """Test the model on a single math problem."""
    # Find the first math problem
    math_problems = [q for q in dataset if q["subject"] == "math"]
    if not math_problems:
        print("No math problems found in dataset!")
        return
    
    question = math_problems[0]
    print("\n=== Testing Single Math Problem ===")
    print(f"\nQuestion Type: {question['type']}")
    print(f"Question Text: {question['question']}")
    print(f"Expected Answer: {question['gold']}")
    
    # Prepare input
    query = MathQuery(
        question=question["question"],
        subject="math",
        type=question["type"]
    )
    
    # Generate solution
    print("\nGenerating solution...")
    solution = model.solve(query)
    
    # Clean and compare answers
    predicted_answer = clean_answer(solution.get("answer", ""), question["type"])
    correct_answer = clean_answer(question["gold"], question["type"])
    
    print("\n=== Results ===")
    print(f"Raw model output: {solution.get('answer', '')}")
    print(f"Cleaned predicted answer: {predicted_answer}")
    print(f"Raw correct answer: {question['gold']}")
    print(f"Cleaned correct answer: {correct_answer}")
    print(f"Match status: {'✓ CORRECT' if predicted_answer == correct_answer else '✗ INCORRECT'}")
    
    print("\n=== Full Solution ===")
    print(solution.get("solution", "No solution provided"))

def main():
    # Initialize components
    print("Initializing Math Professor system...")
    knowledge_base = KnowledgeBase()
    web_searcher = WebSearcher()
    solution_generator = SolutionGenerator()
    feedback_manager = FeedbackManager()
    
    # Load JEE dataset
    print("Loading JEE benchmark dataset...")
    dataset = load_jee_dataset()
    
    # Test single problem
    test_single_problem(dataset, solution_generator)

if __name__ == "__main__":
    main() 