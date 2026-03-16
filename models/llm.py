import os
from langchain_groq import ChatGroq


def get_chatgroq_model():

    try:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        model = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant"
        )

        return model

    except Exception as e:

        raise RuntimeError(f"Groq model initialization failed: {str(e)}")