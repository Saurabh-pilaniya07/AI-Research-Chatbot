import os

# API KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en"
# Data folder
DATA_PATH = "data/documents"