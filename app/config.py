import os

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

assert LANGSMITH_API_KEY, "LANGSMITH_API_KEY is not set!"
assert TAVILY_API_KEY, "TAVILY_API_KEY is not set!"
assert GROQ_API_KEY, "GROQ_API_KEY is not set!"
