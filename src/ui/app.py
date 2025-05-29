import streamlit as st
import sys
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import time
from typing import Dict
import dspy

# Load environment variables
load_dotenv()

# Map environment variables if needed
if os.getenv("TAVILY_AI_KEY") and not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_AI_KEY")

# Verify required API keys
required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.validation.schema import MathQuery
from src.knowledge_base.vectorstore import MathKnowledgeBase
from src.web_search.searcher import MathWebSearcher
from src.llm.solution_generator import MathSolutionGenerator
from src.feedback.collector import FeedbackCollector, FeedbackCategory

def initialize_components():
    """Initialize components with status messages."""
    try:
        # Initialize knowledge base
        st.info("📚 Initializing knowledge base...")
        with st.spinner("Loading Berkeley MATH dataset... This might take a minute or two."):
            kb = MathKnowledgeBase()
        
        # Initialize other components
        st.info("🌐 Setting up web search capabilities...")
        web_searcher = MathWebSearcher()
        
        st.info("🧮 Initializing solution generator...")
        solution_generator = MathSolutionGenerator()
        
        st.info("📝 Setting up feedback system...")
        feedback_collector = FeedbackCollector()
        
        st.success("✅ Setup complete!")
        time.sleep(1)  # Show completion message briefly
        
        return {
            "kb": kb,
            "web_searcher": web_searcher,
            "solution_generator": solution_generator,
            "feedback_collector": feedback_collector
        }
    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        raise e

def render_feedback_section(question: str, solution: str, context_used: bool) -> None:
    """Render the 4Cs feedback section."""
    st.write("### How would you rate this solution?")
    st.write("Please rate each aspect from 1 (needs improvement) to 5 (excellent):")
    
    # Initialize feedback state if not exists
    if 'feedback_ratings' not in st.session_state:
        st.session_state.feedback_ratings = {}
    if 'feedback_text' not in st.session_state:
        st.session_state.feedback_text = {}
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    
    # Create columns for the 4Cs
    cols = st.columns(4)
    category_descriptions = {
        FeedbackCategory.CORRECTNESS: "Mathematical accuracy and validity",
        FeedbackCategory.CLARITY: "Clear and understandable explanations",
        FeedbackCategory.COMPLETENESS: "All necessary steps included",
        FeedbackCategory.CONCISENESS: "Efficient and to-the-point"
    }
    
    # Render rating sliders and feedback boxes for each category
    for i, (category, description) in enumerate(category_descriptions.items()):
        with cols[i]:
            st.write(f"**{category.value.title()}**")
            st.write(f"_{description}_")
            
            # Get previous rating if exists
            default_rating = st.session_state.feedback_ratings.get(category, 3)
            
            rating = st.slider(
                f"{category.value} rating",
                1, 5, default_rating,
                key=f"rating_{category.value}_{question[:10]}"  # Make key unique per question
            )
            st.session_state.feedback_ratings[category] = rating
            
            # Get previous feedback if exists
            default_feedback = st.session_state.feedback_text.get(category, "")
            
            feedback = st.text_area(
                f"Specific feedback for {category.value}",
                value=default_feedback,
                key=f"feedback_{category.value}_{question[:10]}", # Make key unique per question
                height=100
            )
            if feedback:
                st.session_state.feedback_text[category] = feedback
    
    # Submit button
    if not st.session_state.feedback_submitted:
        if st.button("Submit Feedback", key=f"submit_4c_feedback_{question[:10]}"):
            # Submit feedback
            st.session_state.components["feedback_collector"].add_feedback(
                question=question,
                solution=solution,
                ratings=st.session_state.feedback_ratings,
                feedback_text=st.session_state.feedback_text,
                context_used=context_used
            )
            
            # If any rating is below 4, use DSPy to improve the solution
            low_ratings = [c for c, r in st.session_state.feedback_ratings.items() if r < 4]
            if low_ratings:
                feedback_text = "\n".join([
                    f"{c.value}: {st.session_state.feedback_text.get(c, 'Needs improvement')}"
                    for c in low_ratings
                ])
                
                # Use DSPy to generate improved solution
                with st.spinner("🔄 Refining solution based on your feedback..."):
                    new_solution = st.session_state.components["solution_generator"].refine_with_dspy(
                        question=question,
                        previous_solution=solution,
                        feedback_categories=low_ratings,
                        feedback_text=feedback_text
                    )
                
                # Update solution in session state
                st.session_state.current_solution = new_solution
                st.session_state.feedback_submitted = True
                
                # Force a rerun to show the new solution
                st.rerun()
            else:
                st.success("Thank you for your feedback! We're glad the solution was helpful!")
                st.session_state.feedback_submitted = True

def main():
    st.set_page_config(
        page_title="Math Professor AI",
        page_icon="🧮",
        layout="wide"
    )
    
    st.title("🧮 Math Professor AI")
    st.write("Ask any math question and get step-by-step solutions!")
    
    # Initialize components if not already done
    if 'components' not in st.session_state:
        st.session_state.components = initialize_components()
    
    # Initialize solution state
    if 'current_solution' not in st.session_state:
        st.session_state.current_solution = None
        st.session_state.current_question = None
        st.session_state.context_used = False
    
    # Input section
    question = st.text_area(
        "Enter your math question:",
        height=100,
        key="question_input"
    )
    
    if st.button("Get Solution"):
        try:
            # Validate input
            query = MathQuery(query=question)
            
            with st.spinner("🧮 Solving your math problem..."):
                # Try knowledge base first
                kb_results = st.session_state.components["kb"].search(question)
                
                if kb_results and any(result.get("similarity_score", 0) > 0.8 for result in kb_results):
                    solution = kb_results[0]["solution"]
                    context_used = False
                else:
                    # Fallback to web search and solution generation
                    context = st.session_state.components["web_searcher"].search(question)
                    solution = st.session_state.components["solution_generator"].generate(
                        query=question,
                        context=context
                    )
                    context_used = True
                
                # Store in session state
                st.session_state.current_solution = solution
                st.session_state.current_question = question
                st.session_state.context_used = context_used
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Display solution and feedback if available
    if st.session_state.current_solution:
        st.subheader("Solution")
        st.write(st.session_state.current_solution)
        render_feedback_section(
            st.session_state.current_question,
            st.session_state.current_solution,
            st.session_state.context_used
        )
        
        # Add Back to Home button
        if st.button("🏠 Back to Home"):
            # Reset all relevant session state variables
            st.session_state.current_solution = None
            st.session_state.current_question = None
            st.session_state.context_used = False
            st.session_state.feedback_submitted = False
            st.session_state.feedback_ratings = {}
            st.session_state.feedback_text = {}
            st.rerun()

if __name__ == "__main__":
    main() 