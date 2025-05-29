from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Step(BaseModel):
    """Model for a single solution step."""
    number: int
    title: str
    explanation: str
    math_work: Optional[str] = None
    intermediate_result: Optional[str] = None

class Solution(BaseModel):
    """Model for a complete math solution."""
    problem: str
    steps: List[Step]
    final_answer: str
    difficulty_level: str = Field(..., pattern='^(basic|intermediate|advanced)$')
    concepts_used: List[str]
    verification_status: bool = False

class SolutionFormatter:
    """Formats math solutions in a standardized way."""
    
    @staticmethod
    def format_step(step: Step) -> str:
        """Format a single solution step."""
        formatted_step = [
            f"**Step {step.number}: {step.title}**\n",
            f"{step.explanation}\n"
        ]
        
        if step.math_work:
            formatted_step.append(f"```math\n{step.math_work}\n```\n")
            
        if step.intermediate_result:
            formatted_step.append(f"*Intermediate Result:* {step.intermediate_result}\n")
            
        return "\n".join(formatted_step)

    @staticmethod
    def format_solution(solution: Solution) -> str:
        """Format a complete solution with all steps."""
        formatted_solution = [
            "# Math Problem Solution\n",
            f"**Problem Statement:**\n{solution.problem}\n",
            "\n## Solution Steps\n"
        ]
        
        # Add each step
        for step in solution.steps:
            formatted_solution.append(SolutionFormatter.format_step(step))
            
        # Add final answer and metadata
        formatted_solution.extend([
            "\n## Final Answer\n",
            f"**{solution.final_answer}**\n",
            "\n## Solution Metadata\n",
            f"- **Difficulty Level:** {solution.difficulty_level}",
            "- **Key Concepts Used:**",
            "  - " + "\n  - ".join(solution.concepts_used),
            f"\n- **Verification Status:** {'✅ Verified' if solution.verification_status else '❌ Not Verified'}"
        ])
        
        return "\n".join(formatted_solution)

    @staticmethod
    def create_solution_template() -> Dict:
        """Create an empty solution template."""
        return {
            "problem": "",
            "steps": [
                {
                    "number": 1,
                    "title": "Understand the Problem",
                    "explanation": "",
                    "math_work": None,
                    "intermediate_result": None
                },
                {
                    "number": 2,
                    "title": "Plan the Solution",
                    "explanation": "",
                    "math_work": None,
                    "intermediate_result": None
                },
                {
                    "number": 3,
                    "title": "Execute the Plan",
                    "explanation": "",
                    "math_work": None,
                    "intermediate_result": None
                },
                {
                    "number": 4,
                    "title": "Verify the Solution",
                    "explanation": "",
                    "math_work": None,
                    "intermediate_result": None
                }
            ],
            "final_answer": "",
            "difficulty_level": "intermediate",
            "concepts_used": [],
            "verification_status": False
        } 