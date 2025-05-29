from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()

def generate_math_problems():
    """Generate a diverse set of math problems using GPT-4."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    categories = [
        "Calculus (derivatives, integrals, limits)",
        "Algebra (equations, inequalities, functions)",
        "Trigonometry (identities, equations)",
        "Linear Algebra (matrices, vectors)",
        "Statistics (probability, distributions)",
        "Number Theory (primes, divisibility)",
        "Geometry (shapes, proofs)",
    ]
    
    problems = []
    
    for category in categories:
        prompt = f"""Generate 5 math problems in {category} with these requirements:
1. Varying difficulty levels (2 easy, 2 medium, 1 hard)
2. Clear problem statements
3. Detailed step-by-step solutions
4. Include common pitfalls to avoid

Format each problem as a JSON object with:
{{
    "question": "problem statement",
    "solution": "detailed step-by-step solution",
    "category": "math category",
    "difficulty": "easy/medium/hard",
    "concepts": ["list", "of", "concepts", "tested"]
}}

Return ONLY a JSON array of 5 problems, nothing else. Start with [ and end with ]."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert math professor creating practice problems. Always return valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        try:
            # Parse the JSON response
            category_problems = json.loads(response.choices[0].message.content)
            problems.extend(category_problems)
            print(f"✅ Generated {len(category_problems)} problems for {category}")
        except Exception as e:
            print(f"❌ Error generating problems for {category}: {e}")
            print(f"Response was: {response.choices[0].message.content[:200]}...")
    
    # Save to file
    output_file = "data/math_qa_expanded.json"
    os.makedirs("data", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(problems, f, indent=4)
    
    print(f"\n✨ Generated {len(problems)} total problems and saved to {output_file}")

if __name__ == "__main__":
    generate_math_problems() 