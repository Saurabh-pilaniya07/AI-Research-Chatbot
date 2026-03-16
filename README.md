# Intelligent AI Research Chatbot

A hybrid AI assistant built with **Streamlit, LangChain, Groq LLM, and FAISS** that combines **Retrieval-Augmented Generation (RAG)** with **live web search** to generate grounded answers with transparent sources.

The system can answer questions using:

* Uploaded documents (PDF/TXT)
* Live internet search
* LLM reasoning

This architecture resembles modern AI research assistants such as **Perplexity AI**.

---

# Features

### Conversational Chat Interface

Interactive chatbot built with **Streamlit**.

### Document Question Answering

Upload **PDF or TXT documents** and ask questions about them.

### Retrieval-Augmented Generation (RAG)

Documents are:

1. Split into chunks
2. Converted into embeddings
3. Stored in a FAISS vector database
4. Retrieved via semantic similarity search

### Web Search Integration

If the question is unrelated to uploaded documents, the system performs **live web search** using SerpAPI.

### LLM Tool Routing

The LLM decides whether the query should use:

* Document Retrieval (RAG)
* Web Search

This enables more intelligent behavior compared to rule-based routing.

### Source Transparency

The chatbot displays:

* Source links
* Retrieved context

Similar to research assistants like Perplexity.

### Response Modes

Users can switch between:

Concise mode

* Short responses
* 3–5 sentences

Detailed mode

* Structured explanations
* More context

---

# System Architecture

```
User Query
    ↓
Streamlit Interface
    ↓
LLM Tool Router
    ↓
Decision Layer
   ├── Document Retrieval (FAISS)
   └── Web Search (SerpAPI)
    ↓
Context Builder
    ↓
Groq LLM
    ↓
Answer + Sources
```

---

# RAG Pipeline

```
Documents (PDF/TXT)
        ↓
Text Chunking
        ↓
Embeddings (Sentence Transformers)
        ↓
FAISS Vector Database
        ↓
Similarity Search
        ↓
Relevant Context
        ↓
LLM Answer Generation
```

---

# Usage

### Ask General Questions

Example:

```
Latest AI trends in 2026
```

The system will use **web search**.

---

### Upload a Document

1. Navigate to **Upload File**
2. Upload a PDF or TXT
3. Switch to **Chat**

Example question:

```
Summarize the main findings of the document
```

The system will use **RAG retrieval**.

---

# Technologies Used

Core Stack

* Python
* Streamlit

AI & NLP

* LangChain
* Groq LLM
* HuggingFace Embeddings
* Sentence Transformers

Retrieval

* FAISS Vector Database

External Data

* SerpAPI

---

# Example Workflow

1. User asks a question.
2. LLM decides which tool to use.
3. If document-related → FAISS retrieval.
4. If general knowledge → Web search.
5. Context is sent to Groq LLM.
6. Model generates grounded response.
7. Sources are displayed.

---

# Future Improvements

Potential extensions include:

* Conversation memory
* Multi-document indexing
* Streaming responses
* Advanced tool agents
* Better citation ranking
* Vector database persistence