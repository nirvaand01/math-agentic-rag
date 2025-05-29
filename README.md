# Math Professor AI

An AI-powered math tutor that provides step-by-step solutions to mathematical problems using the Berkeley MATH dataset and GPT-4.

## Features

- Step-by-step math problem solutions
- Integration with Berkeley MATH dataset
- Web search capabilities for additional context
- Feedback system with 4C framework (Correctness, Clarity, Completeness, Conciseness)
- Adaptive solutions based on user feedback

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd math-agentic-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
SERPER_API_KEY=your_serper_api_key
```

4. Run the app:
```bash
streamlit run src/ui/app.py
```

## Deployment

This app can be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your environment variables in the Streamlit Cloud dashboard
5. Deploy!

## Environment Variables

The following environment variables are required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key for web search
- `SERPER_API_KEY`: Your Serper API key for web search

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 