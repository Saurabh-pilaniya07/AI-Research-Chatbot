from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL


def get_embedding_model():
    try:

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        return embeddings

    except Exception as e:

        raise RuntimeError(f"Embedding model error: {str(e)}")