from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

class MCPSolutionVerifier:
    def __init__(self):
        """Initialize the MCP solution verifier."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def break_down_problem(self, query: str) -> List[Dict]:
        """Break down a complex math problem into smaller, verifiable steps."""
        system_prompt = """You are a math expert that breaks down complex problems into smaller, verifiable steps.
For each step, you should:
1. State what needs to be solved
2. Explain why this step is necessary
3. List any prerequisites needed
4. Describe how to verify the result
5. Identify any potential edge cases

Format your response as a series of steps, with each step containing:
Step 1:
- Task: [what needs to be solved]
- Explanation: [why this step is necessary]
- Prerequisites: [list of prerequisites]
- Verification: [how to verify the result]
- Edge cases: [potential edge cases to consider]

Step 2:
[and so on...]"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Break down this math problem: {query}"}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        # Parse the response
        try:
            steps_text = response.choices[0].message.content
            steps_list = []
            current_step = None
            
            for line in steps_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.lower().startswith('step'):
                    if current_step:
                        steps_list.append(current_step)
                    current_step = {
                        "task": "",
                        "explanation": "",
                        "prerequisites": [],
                        "verification": "",
                        "edge_cases": []
                    }
                elif current_step is not None:
                    if line.startswith('- Task:') or line.startswith('Task:'):
                        current_step["task"] = line.split(':', 1)[1].strip()
                    elif line.startswith('- Explanation:') or line.startswith('Explanation:'):
                        current_step["explanation"] = line.split(':', 1)[1].strip()
                    elif line.startswith('- Prerequisites:') or line.startswith('Prerequisites:'):
                        prereqs = line.split(':', 1)[1].strip()
                        current_step["prerequisites"] = [p.strip() for p in prereqs.split(',') if p.strip()]
                    elif line.startswith('- Verification:') or line.startswith('Verification:'):
                        current_step["verification"] = line.split(':', 1)[1].strip()
                    elif line.startswith('- Edge cases:') or line.startswith('Edge cases:'):
                        edge_cases = line.split(':', 1)[1].strip()
                        current_step["edge_cases"] = [e.strip() for e in edge_cases.split(',') if e.strip()]
            
            if current_step:
                steps_list.append(current_step)
            
            # Validate steps
            valid_steps = []
            for step in steps_list:
                if step["task"] and step["explanation"]:  # Minimum requirements
                    valid_steps.append(step)
            
            return valid_steps if valid_steps else []
            
        except Exception as e:
            print(f"Error parsing steps: {e}")
            print(f"Raw response: {steps_text}")
            return []

    def verify_step(self, step: Dict, solution: str) -> bool:
        """Verify if a solution step is correct."""
        prompt = f"""Verify if this solution step is correct:

Step to verify: {step['task']}
Proposed solution: {solution}
Verification method: {step['verification']}
Edge cases to consider: {', '.join(step['edge_cases'])}

Is the solution correct? Respond with 'Yes' or 'No' and explain why."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a math expert that verifies solution steps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )

        verification_result = response.choices[0].message.content.lower()
        return verification_result.startswith('yes')

    def generate_solution(self, step: Dict) -> str:
        """Generate a solution for a single step."""
        prompt = f"""Solve this math step:

Task: {step['task']}
Prerequisites: {', '.join(step['prerequisites'])}
Verification method: {step['verification']}
Edge cases to consider: {', '.join(step['edge_cases'])}

Provide a clear, step-by-step solution."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a math expert that provides clear, step-by-step solutions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content

    def generate_solution_with_verification(self, query: str) -> str:
        """Generate and verify a complete solution for a math problem."""
        # Break down the problem into steps
        steps = self.break_down_problem(query)
        if not steps:
            return "Could not break down the problem into steps."

        # Generate and verify solution for each step
        complete_solution = []
        for i, step in enumerate(steps, 1):
            solution = self.generate_solution(step)
            is_correct = self.verify_step(step, solution)
            
            if not is_correct:
                # Try one more time with more detailed instructions
                solution = self.generate_solution({
                    **step,
                    "task": f"Carefully solve: {step['task']}",
                    "prerequisites": step['prerequisites'] + ["Double-check each calculation"]
                })
                is_correct = self.verify_step(step, solution)
                
                if not is_correct:
                    return f"Could not generate a verified solution for step {i}: {step['task']}"
            
            complete_solution.append(f"Step {i}: {step['task']}\n{solution}\n")

        return "\n".join(complete_solution) 