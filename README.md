# 🏠 Intelligent Property Price Prediction (with GenAI)

**Project 9 — Capstone Project**

A modular machine learning system that predicts Melbourne property prices using classical ML models (Scikit-Learn), enhanced by an **Agentic AI & GenAI RAG Application** built with LangChain.

## 🌟 New Agentic Features
This project now features an integrated AI Assistant that can:
- **RAG Capability**: Perform vector similarity search using FAISS against the Capstone Report (`report/GenAI Capstone Project.pdf`).
- **LangChain Tool Calling**: The agent acts as an autonomous assistant using `create_tool_calling_agent`.
- **Model as a Tool**: The agent is equipped with a `predict_property_price` tool. It can seamlessly extract property features from natural language and query the local Scikit-Learn pipeline to estimate property prices.

---

## 📁 Project Structure

```
GenAI/
├── data/
│   ├── melbourne_housing.csv    # Dataset (13,581 rows)
│   └── faiss_index/             # Auto-generated Vector Store (RAG)
├── models/
│   ├── best_model.pkl           # Trained model pipeline (auto-generated)
│   └── model_metadata.json      # Feature list & metrics (auto-generated)
├── report/
│   └── GenAI Capstone Project.pdf # Reference PDF for Vector Search
├── rag_agent.py                 # LangChain Agent and Tool logic
├── data_preprocessing.py        # Data loading, feature engineering, sklearn pipeline
├── train_model.py               # Model training, evaluation & persistence
├── app.py                       # Streamlit web application & AI Chat
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

You must install traditional ML tools alongside GenAI capabilities.

```bash
pip install -r requirements.txt
```

### 2. Setup your Environment

Copy `.env.example` to a `.env` file and insert your keys:

```bash
cp .env.example .env
```

Preferred setup:
- `POLLINATIONS_API_KEY` (primary)
- `OPENAI_API_KEYS` (fallback rotation, optional)

This project may also use `GROQ_API_KEY` / `GEMINI_API_KEY` depending on the agent configuration in `rag_agent.py`.

### 3. Train the Model (Optional)

If `models/best_model.pkl` doesn't exist, train the baseline model:

```bash
python train_model.py
```

### 4. Launch the Web App

Start the Streamlit application. **Note: During the first launch, the Capstone PDF will be processed automatically and cached into a local FAISS database for lightning-fast retrieval.**

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. You can flip between the **Manual Prediction Dashboard** and the **AI Real Estate Agent** tab!

---

## 🔬 Methodology

### Agent Framework
- **LLM**: Gemini 1.5 Flash (via `ChatGoogleGenerativeAI`) or GPT-3.5-Turbo (auto-detected via API Keys). 
- **Embeddings**: Google Generative AI Embeddings or OpenAI Embeddings.
- **Vector Database**: FAISS (Facebook AI Similarity Search) optimized for local CPU usage without external server dependencies.

### Scikit-Learn Pipeline
| Step | Numerical Features | Categorical Features |
|------|-------------------|---------------------|
| **Imputation** | `SimpleImputer(strategy='median')` | `SimpleImputer(strategy='most_frequent')` |
| **Transformation** | `StandardScaler` | `OneHotEncoder(handle_unknown='ignore')` |

---

## 🛠️ Tech Stack

- **LangChain** — LLMs, Prompts, Agents, Tools, RAG
- **FAISS** — Local Vector Search indexing
- **PyPDF** — Document parsing
- **Scikit-Learn** — ML pipelines, models, preprocessing
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Interactive UI tabs and Chat
- **Joblib** — Model serialization

---

## 📄 License

This project is for educational purposes as part of a Gen AI Capstone project, complying strictly with Agentic AI and GenAI Evaluation Criteria.
