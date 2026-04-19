"""
rag_agent.py
============
Agentic AI and RAG module for the Melbourne Property Price Predictor.
Implements a LangChain Tool Calling Agent equipped with RAG over the report PDF
and a Scikit-Learn Model Prediction wrapper tool.
"""

import os
import joblib
import pandas as pd
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()


# Paths and Configurations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "report", "final_report.md")
FAISS_DB_PATH = os.path.join(BASE_DIR, "data", "faiss_index")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")


# LLM and Embeddings Initialization

def get_llm():
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    elif os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    else:
        raise ValueError("No API key found. Please set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your .env file.")

def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # Using local embeddings so we totally bypass Google's embedding REST API blocks
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Vector Store (RAG)

def initialize_vector_store():
    """Load Markdown, chunk it, and save to FAISS, or load existing FAISS index."""
    embeddings = get_embeddings()
    if os.path.exists(FAISS_DB_PATH):
        print("Loading existing FAISS vector store from disk...")
        return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("Ingesting Capstone Markdown Report to create FAISS vector store...")
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"Markdown Report not found at: {REPORT_PATH}")

    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(REPORT_PATH)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DB_PATH)
    return vectorstore


# Agent Tools

@tool
def predict_property_price(
    rooms: int, distance: float, bedroom2: int, bathroom: int, 
    car: int, landsize: float, building_area: float, 
    house_age: int, property_type: str, regionname: str
) -> str:
    """Predicts a Melbourne property price using the trained machine learning model.
    Provide the features: rooms, distance, bedroom2, bathroom, car, landsize, 
    building_area, house_age, property_type, and regionname.
    
    property_type must be one of: 'h' (house), 't' (townhouse), 'u' (unit).
    regionname must be one of: 'Northern Metropolitan', 'Western Metropolitan', 'Southern Metropolitan', 
    'Eastern Metropolitan', 'South-Eastern Metropolitan', 'Northern Victoria', 'Western Victoria', 'Eastern Victoria'.
    """
    try:
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([{
            "Rooms": rooms,
            "Distance": distance,
            "Bedroom2": bedroom2,
            "Bathroom": bathroom,
            "Car": car,
            "Landsize": landsize,
            "BuildingArea": building_area,
            "HouseAge": house_age,
            "Type": property_type,
            "Regionname": regionname,
        }])
        pred = model.predict(input_df)[0]
        return f"The predicted property price is ${max(pred, 0):,.0f} AUD."
    except Exception as e:
        return f"Error during prediction: {str(e)}"


# Agent Initialization

def create_agent():
    """Initializes and returns the LangChain AgentExecutor."""
    llm = get_llm()
    vectorstore = initialize_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    @tool
    def search_project_report(query: str) -> str:
        """Searches and returns excerpts from the Intelligent Property Price Prediction report. Use this to answer context rules, architectural questions, or details about the project."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_tool = search_project_report
    
    @tool
    def search_similar_properties(query_location: str, max_price: float = None, min_rooms: int = None) -> str:
        """Searches the original Melbourne housing dataset to recommend existing real-world properties matching user criteria.
        Args:
            query_location: A suburb or region name to search for (e.g. 'Northern Metropolitan', 'Reservoir').
            max_price: Optional maximum price ceiling.
            min_rooms: Optional minimum number of rooms needed.
        Returns:
            JSON string summary of the best matching real properties if found.
        """
        try:
            import pandas as pd
            df = pd.read_csv("data/melbourne_housing.csv")
            
            # Filter by location (Regionname OR Suburb)
            mask = df["Regionname"].str.contains(query_location, case=False, na=False) | df["Suburb"].str.contains(query_location, case=False, na=False)
            df = df[mask]
            
            if df.empty:
                return f"No properties found in location matching '{query_location}'."
                
            df = df.dropna(subset=['Price'])
            if max_price:
                df = df[df['Price'] <= max_price]
            if min_rooms:
                df = df[df['Rooms'] >= min_rooms]
                
            if df.empty:
                return f"No properties found matching location: '{query_location}', max_price: {max_price}, min_rooms: {min_rooms}."
                
            # Get top matches, sorted by price.
            df = df.sort_values(by='Price', ascending=True)
            results = df.head(5)[['Suburb', 'Rooms', 'Price', 'Type', 'Regionname']].to_dict('records')
            return f"Found {len(df)} properties matching criteria. Top matches: {results}"
        except Exception as e:
            return f"Error searching dataset: {str(e)}"
            
    dataset_search_tool = search_similar_properties
    
    tools = [rag_tool, predict_property_price, dataset_search_tool]
    
    system_prompt = """You are an intelligent, helpful AI real estate assistant equipped with 3 tools:
1. `search_project_report`: To answer questions about the project's prediction architecture, testing methodology, mathematical formulas (like MAE), dataset engineering (like HouseAge), and evaluation metrics (like R2 scores). You MUST use this tool to retrieve precise facts from the PDF report. Do not hallucinate math or scores.
2. `predict_property_price`: To predict the price of a *hypothetical* property. You MUST have all 10 parameters to use this.
3. `search_similar_properties`: To find existing real-world properties in the dataset (e.g. "Find me a property", "Can I get a better property", or "Show me houses in X"). Use this primarily.

CRITICAL INSTRUCTIONS:
- If a user asks about the report, methodology, model scores, architecture, or equations, immediately call `search_project_report` and cite the results.
- If a user asks to "find", "get", or "show" properties, use `search_similar_properties` immediately. If you are missing a location or parameters to search, just cleanly and casually ask "What region or suburb?"—do NOT dump a long bulleted error or list.
- Only ever ask for the 10 specific ML prediction parameters if the user explicitly says they want to "predict" a custom house."""
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent
