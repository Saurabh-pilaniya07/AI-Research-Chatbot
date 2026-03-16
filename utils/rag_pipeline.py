import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.embeddings import get_embedding_model
from config.config import DATA_PATH


@st.cache_resource
def create_vector_store():

    documents = []

    for file in os.listdir(DATA_PATH):

        path = os.path.join(DATA_PATH, file)

        loader = TextLoader(path)

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


def retrieve_context(query):

    try:

        store = create_vector_store()

        docs = store.similarity_search(query, k=3)

        return "\n".join([doc.page_content for doc in docs])

    except Exception:
        return ""