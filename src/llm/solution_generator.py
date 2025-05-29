from typing import Dict, Optional, Any, List
from openai import OpenAI
from dotenv import load_dotenv
import os
import dspy
import asyncio
import nest_asyncio
from src.validation.schema import MathQuery
from src.knowledge_base.vectorstore import MathKnowledgeBase
from src.web_search.searcher import MathWebSearcher
from src.feedback.collector import FeedbackCategory

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

load_dotenv()

# Configure DSPy with GPT-4
lm = dspy.LM(
    "openai/gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_type="chat"
)
dspy.settings.configure(lm=lm)

class MathSolutionRefinementSignature(dspy.Signature):
    """Signature for refining math solutions based on feedback."""
    
    question = dspy.InputField(desc="The original math question")
    previous_solution = dspy.InputField(desc="The previous solution that needs improvement")
    feedback = dspy.InputField(desc="Specific feedback on what needs improvement")
    categories = dspy.InputField(desc="Categories that need improvement (correctness, clarity, completeness, conciseness)")
    solution = dspy.OutputField(desc="The improved solution addressing the feedback")

class MathSolutionRefiner(dspy.Module):
    """DSPy module for refining math solutions based on feedback."""
    
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(MathSolutionRefinementSignature)
    
    def forward(self, question: str, previous_solution: str, feedback: str, categories: List[str]) -> str:
        """Refine the solution based on feedback."""
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = self.refine(
                question=question,
                previous_solution=previous_solution,
                feedback=feedback,
                categories=", ".join(categories)
            )
            return result.solution
        except Exception as e:
            print(f"Error in DSPy refinement: {str(e)}")
            # Fallback to regular OpenAI completion if DSPy fails
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"""Given a math problem and feedback, provide an improved solution.

Question: {question}

Previous Solution:
{previous_solution}

Feedback: {feedback}
Categories needing improvement: {", ".join(categories)}

Please provide an improved solution addressing the feedback."""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert math professor improving a solution based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

class SolutionGenerator:
    """Class to generate solutions for math problems using GPT-4."""
    
    def __init__(self):
        """Initialize the solution generator."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.knowledge_base = MathKnowledgeBase()
        self.web_searcher = MathWebSearcher()
        
        # Load prompt templates for answer extraction
        self.answer_templates = {
            "MCQ": """Solve this multiple choice math problem step by step to find the correct answer:

Here's a similar example:
Q: Let -π/4 < θ < 0. If α and β are roots of x² + 2x tan θ + 1 = 0 and α > β, find α + β.
A: Since -π/4 < θ < 0, tan θ is negative.
For x² + 2x tan θ + 1 = 0:
a = 1, b = 2 tan θ, c = 1
Sum of roots = -b/a = -2 tan θ
Therefore, α + β = -2 tan θ

Now solve your problem:

{question}

Remember:
1. For -π/6 < θ < -π/12:
   - sin θ is negative
   - cos θ is positive
   - tan θ is negative
   - sec θ is positive

2. For quadratic ax² + bx + c = 0:
   - Sum of roots = -b/a
   - Product of roots = c/a

Provide ONLY the letter (A/B/C/D) that matches your final calculation.

Answer:""",
            "MCQ(multiple)": """Read the following math problem and provide ONLY the combination of correct answer letters (e.g., ABC). Do not explain.

{question}

Answer:""",
            "Integer": """Read the following math problem and provide ONLY the final integer answer. Do not explain.

{question}

Answer:""",
            "Numeric": """Read the following math problem and provide ONLY the final numerical answer. Do not explain.

{question}

Answer:"""
        }
        
        # Load prompt templates for explanations
        self.explanation_templates = {
            "MCQ": "Please solve this multiple choice math problem step by step:\n{question}\n\nProvide your answer as a single letter (A/B/C/D) at the end.",
            "MCQ(multiple)": "Please solve this multiple choice math problem step by step:\n{question}\n\nProvide your answer as a combination of letters (e.g., 'ABC') at the end.",
            "Integer": "Please solve this math problem step by step:\n{question}\n\nProvide your final answer as a single integer at the end.",
            "Numeric": "Please solve this math problem step by step:\n{question}\n\nProvide your final answer as a number at the end."
        }
    
    def solve(self, query: MathQuery) -> Dict[str, Any]:
        """
        Generate solution for a math problem.
        
        Args:
            query: MathQuery object containing the question and metadata
            
        Returns:
            Dictionary containing the solution and answer
        """
        # Get appropriate prompt template
        prompt_template = self.explanation_templates.get(query.type, self.explanation_templates["MCQ"])
        
        # Format prompt
        prompt = prompt_template.format(question=query.question)
        
        try:
            # Call GPT-4 to generate solution
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a highly skilled mathematics professor who excels at solving complex math problems step by step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Extract solution text
            solution_text = response.choices[0].message.content
            
            # Extract final answer
            answer_lines = [line for line in solution_text.split("\n") if line.strip()]
            final_answer = answer_lines[-1].strip()
            
            # For MCQ and MCQ(multiple), extract just the letter(s)
            if query.type in ["MCQ", "MCQ(multiple)"]:
                final_answer = ''.join(c for c in final_answer if c.isalpha()).upper()
            # For Integer and Numeric, extract just the number
            elif query.type in ["Integer", "Numeric"]:
                final_answer = ''.join(c for c in final_answer if c.isdigit() or c in ['.', '-'])
            
            return {
                "solution": solution_text,
                "answer": final_answer
            }
            
        except Exception as e:
            print(f"Error generating solution: {str(e)}")
            return {
                "solution": "",
                "answer": ""
            }

    def get_answer(self, query: MathQuery, kb_context: str, web_context: str) -> str:
        """Get just the answer without explanation."""
        prompt = self.answer_templates[query.type].format(
            question=query.question,
            kb_context=kb_context,
            web_context=web_context
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise math problem solver. For trigonometric problems in the domain -π/6 < θ < -π/12, remember: sin θ and tan θ are negative, cos θ and sec θ are positive. Provide only the answer without explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting answer: {str(e)}")
            return ""

class MathSolutionGenerator:
    def __init__(self):
        """Initialize the solution generator with OpenAI and DSPy."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.refiner = MathSolutionRefiner()
        self.knowledge_base = MathKnowledgeBase()
        self.web_searcher = MathWebSearcher()
        
        self.base_prompt = """You are an expert math professor who explains solutions step by step.
Given a math problem and relevant context, provide a clear, student-friendly solution.
Break down the solution into clear steps and explain the reasoning behind each step.

Format your response as follows:
1. First restate the problem
2. List any key concepts or formulas needed
3. Provide the solution steps with explanations
4. Give a final answer
5. Add a brief note about common mistakes to avoid

Remember to:
- Use clear mathematical notation
- Explain each step's reasoning
- Point out key insights
- Highlight common pitfalls
"""

    def generate(self, query: str, context: Optional[str] = None) -> str:
        """Generate a step-by-step solution for a math problem."""
        prompt = self.base_prompt + f"\n\nProblem: {query}"
        if context:
            prompt += f"\n\nRelevant Context: {context}"
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please solve this problem step by step."}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def regenerate_with_feedback(
        self,
        question: str,
        previous_solution: str,
        feedback: str,
        context: Optional[str] = None
    ) -> Dict:
        """Regenerate solution incorporating user feedback."""
        prompt = self.base_prompt + f"""
\nQuestion: {question}

Previous Solution:
{previous_solution}

User Feedback:
{feedback}

Please provide an improved solution addressing the feedback."""

        if context:
            prompt += f"\n\nAdditional Context:\n{context}"
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please provide an improved solution."}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        improved_solution = response.choices[0].message.content
        
        return {
            "question": question,
            "solution": improved_solution,
            "model_used": "gpt-4-turbo-preview",
            "context_used": bool(context),
            "feedback_incorporated": True
        }

    def refine_with_dspy(
        self,
        question: str,
        previous_solution: str,
        feedback_categories: List[FeedbackCategory],
        feedback_text: str
    ) -> str:
        """
        Use DSPy to refine the solution based on feedback.
        
        Args:
            question: Original math question
            previous_solution: Previous solution that needs improvement
            feedback_categories: List of categories that need improvement
            feedback_text: Specific feedback text
            
        Returns:
            Improved solution addressing the feedback
        """
        # Convert feedback categories to strings
        categories = [cat.value for cat in feedback_categories]
        
        # Use DSPy to generate improved solution
        improved_solution = self.refiner(
            question=question,
            previous_solution=previous_solution,
            feedback=feedback_text,
            categories=categories
        )
        
        return improved_solution 