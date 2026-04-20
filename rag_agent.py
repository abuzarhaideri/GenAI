"""
rag_agent.py
============
Agentic AI and RAG module for the Melbourne Property Price Predictor.
Implements a LangChain Tool Calling Agent equipped with RAG over the report PDF
and a Scikit-Learn Model Prediction wrapper tool.
"""

"""
rag_agent.py
============
Deprecated in favor of `ai_chat.py`.

This project now uses Pollinations (no-key endpoint) for chat by default, with an optional
OpenRouter fallback. This module is kept only for backwards compatibility with older imports.
"""


def create_agent():  # pragma: no cover
    raise RuntimeError(
        "This project no longer uses rag_agent.py. The Streamlit app uses ai_chat.py instead."
    )
