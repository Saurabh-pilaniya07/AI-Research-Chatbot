import streamlit as st
import os
import sys
import tempfile
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from models.llm import get_chatgroq_model
from models.embeddings import get_embedding_model
from utils.web_search import search_web


# -------------------------
# FILE PROCESSING (PDF / TXT)
# -------------------------

def process_uploaded_file(uploaded_file):

    try:

        suffix = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        if suffix == "pdf":
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80
        )

        docs = splitter.split_documents(docs)

        embeddings = get_embedding_model()

        vector_store = FAISS.from_documents(docs, embeddings)

        return vector_store

    except Exception as e:

        st.error(f"File processing failed: {e}")
        return None


# -------------------------
# LLM TOOL ROUTER
# Decide between RAG or WEB
# -------------------------

def decide_tool(chat_model, query):

    prompt = """
You are an AI router.

Your task is to decide which tool should answer the user's question.

TOOLS:

RAG
Use this if the question is about an uploaded document.

WEB
Use this if the question requires general knowledge, news, or internet information.

Respond with ONLY ONE WORD:
RAG or WEB
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ]

    try:

        response = chat_model.invoke(messages).content.strip().upper()

        if "RAG" in response:
            return "rag"

        return "web"

    except Exception:

        return "web"


# -------------------------
# CONTEXT RETRIEVAL
# -------------------------

def get_context(chat_model, query):

    try:

        tool = decide_tool(chat_model, query)

        if tool == "rag" and "vector_store" in st.session_state:

            docs = st.session_state.vector_store.similarity_search(query, k=3)

            context = "\n".join([d.page_content for d in docs])

            sources = ["Uploaded document"]

            return context, sources

        # fallback to web search

        context, sources = search_web(query)

        return context, sources

    except Exception:

        return "", []


# -------------------------
# LLM RESPONSE
# -------------------------

def get_chat_response(chat_model, query, system_prompt, context):

    try:

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""
Context:
{context}

Question:
{query}

Answer using ONLY the context above.
"""
            )
        ]

        response = chat_model.invoke(messages)

        return response.content

    except Exception as e:

        return f"Error generating response: {str(e)}"


# -------------------------
# CHAT PAGE
# -------------------------

def chat_page():

    st.title("Intelligent AI Chatbot")

    chat_model = get_chatgroq_model()

    # Sidebar settings
    with st.sidebar:

        st.header("Settings")

        response_mode = st.selectbox(
            "Response Mode",
            ["Concise", "Detailed"]
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # PROMPTS

    if response_mode == "Concise":

        system_prompt = """
You are an AI research assistant.

Rules:
- Answer using ONLY the provided context.
- Do not invent facts.
- If the context lacks information say:
  "I could not find reliable information in the sources."
- Keep answers short (3–5 sentences).

End with:

Sources:
"""

    else:

        system_prompt = """
You are an AI research assistant.

Rules:
- Use ONLY the provided context.
- Do not introduce knowledge outside the context.
- If context is incomplete say so.
- Provide a structured explanation.

End with:

Sources:
"""

    # Chat history

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input

    if prompt := st.chat_input("Ask anything..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Searching..."):

                context, sources = get_context(chat_model, prompt)

                response = get_chat_response(
                    chat_model,
                    prompt,
                    system_prompt,
                    context
                )

                st.markdown(response)

                # SOURCE DISPLAY

                if sources:

                    with st.expander("Sources"):

                        for s in sources:

                            if s.startswith("http"):
                                st.markdown(f"- [{s}]({s})")
                            else:
                                st.write(f"- {s}")

                # CONTEXT VIEWER

                with st.expander("Retrieved Context"):

                    st.write(context)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


# -------------------------
# UPLOAD PAGE
# -------------------------

def upload_page():

    st.title("Upload Document")

    uploaded_file = st.file_uploader(
        "Upload TXT or PDF file",
        type=["txt", "pdf"]
    )

    if uploaded_file:

        st.info("Processing document...")

        st.session_state.vector_store = process_uploaded_file(uploaded_file)

        st.success("Document indexed successfully!")

        st.info("Switch to Chat page to ask questions about the document.")


# -------------------------
# MAIN APP
# -------------------------

def main():

    st.set_page_config(
        page_title="AI Research Chatbot",
        layout="wide"
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Chat", "Upload File"]
    )

    # reset doc if switching back to chat
    if page == "Chat" and "vector_store" not in st.session_state:
        pass

    if page == "Chat":
        chat_page()

    if page == "Upload File":
        upload_page()


if __name__ == "__main__":
    main()