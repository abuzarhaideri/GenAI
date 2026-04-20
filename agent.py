import os
import joblib
import pandas as pd
from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from llm_provider import get_openai_keys_from_env, invoke_with_pollinations_fallback

# ---------------------------------------------------------------------------
# Setup & Model Loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
# Use the uncleaned dataset for market statistics if needed.
DATASET_PATH = os.path.join(BASE_DIR, "data", "melbourne_housing.csv")

try:
    _model = joblib.load(MODEL_PATH)
except Exception:
    _model = None

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def predict_property_price(
    rooms: int, distance: float, bedroom2: int, bathroom: int, 
    car: int, landsize: float, building_area: float, 
    house_age: int, property_type: str, regionname: str
) -> str:
    """
    Predicts the precise estimated price of a property in Melbourne based on its specific characteristics.
    Always use this tool when the user asks for a price estimate for a property.
    
    Args:
        rooms: Total number of rooms
        distance: Distance from Melbourne CBD in kilometers (km)
        bedroom2: Number of bedrooms (usually similar to rooms)
        bathroom: Number of bathrooms
        car: Number of parking/car spaces
        landsize: Land size in square meters (m2)
        building_area: Building interior area in square meters (m2)
        house_age: Age of the property in years
        property_type: Input MUST be exactly 'h' (house,cottage,villa, semi,terrace), 'u' (unit, duplex), or 't' (townhouse).
        regionname: The region name (e.g. 'Northern Metropolitan', 'Southern Metropolitan', 'Eastern Metropolitan', 'Western Metropolitan', 'South-Eastern Metropolitan', 'Eastern Victoria', 'Northern Victoria', 'Western Victoria').
    """
    if _model is None:
        return "Error: Machine Learning model not loaded. Please ensure the model is trained."
    
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
    
    try:
        prediction = _model.predict(input_df)[0]
        prediction = max(prediction, 0)
        return f"Tool Output: The predicted property price is ${prediction:,.0f} AUD. Please present this nicely to the user."
    except Exception as e:
        return f"Tool Error predicting price: {str(e)}"

@tool
def get_market_statistics(regionname: str) -> str:
    """
    Retrieves aggregated market statistics for a specific region in Melbourne.
    Use this to give macro-level context or answer questions about general market trends in a region.
    Args:
        regionname: The specific region to query (e.g., 'Southern Metropolitan', 'Northern Metropolitan').
    """
    if not os.path.exists(DATASET_PATH):
        return f"Tool Error: Unable to fetch statistics. Dataset not found at {DATASET_PATH}."
    
    try:
        df = pd.read_csv(DATASET_PATH)
        region_data = df[df['Regionname'] == regionname]
        if region_data.empty:
            return f"No historical market data found for region: {regionname}. Please check the region name."
        
        avg_price = region_data['Price'].mean()
        avg_distance = region_data['Distance'].mean()
        count = len(region_data)
        
        return (f"Market Statistics for {regionname}:\n"
                f"- Number of historical property sales on record: {count}\n"
                f"- Average Sale Price: ${avg_price:,.0f} AUD\n"
                f"- Average Distance to CBD: {avg_distance:.1f} km\n")
    except Exception as e:
        return f"Error computing market statistics: {str(e)}"

# ---------------------------------------------------------------------------
# RAG Setup & Tool
# ---------------------------------------------------------------------------
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    
    openai_keys = get_openai_keys_from_env()
    if not openai_keys:
        return None
        
    embeddings = OpenAIEmbeddings(api_key=openai_keys[0], model="text-embedding-3-small")
    if os.path.exists(VECTOR_DB_PATH):
        try:
            _vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            return _vector_store
        except Exception:
            pass # fallback to recreate
    
    docs = []
    report_dir = os.path.join(BASE_DIR, "report")
    if os.path.exists(report_dir):
        for root, _, files in os.walk(report_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        docs.append(Document(page_content=f.read(), metadata={"source": file}))
    
    if not docs:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    _vector_store = FAISS.from_documents(splits, embeddings)
    _vector_store.save_local(VECTOR_DB_PATH)
    return _vector_store

@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the project's knowledge base (documentation, mathematical methodology, metrics) 
    using Retrieval-Augmented Generation (RAG). 
    Use this to answer questions about the project's architecture, models, performance, or methodology.
    """
    vs = get_vector_store()
    if vs is None:
        return (
            "RAG knowledge base is disabled (no embeddings provider configured). "
            "Configure embeddings to enable knowledge base search."
        )
    docs = vs.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n".join([f"Source ({d.metadata.get('source', 'unknown')}):\n{d.page_content}" for d in docs])

# Define the tools array
tools = [predict_property_price, get_market_statistics, search_knowledge_base]

# ---------------------------------------------------------------------------
# LangGraph Workflow Definition
# ---------------------------------------------------------------------------
def build_langgraph_agent():
    """
    Constructs and returns the LangGraph application.
    """
    # 2. System prompt
    sys_msg = SystemMessage(content=(
        "You are 'Melbourne EstateAI', an expert real estate agent and predictive assistant built with LangGraph. "
        "You can answer user questions about Melbourne real estate, perform price predictions, AND answer queries about your own system architecture, ML models, evaluation metrics (R2, MAE, RMSE), and methodology. "
        "If a user asks about how you work, use the search_knowledge_base tool to retrieve accurate information from the project reports. "
        "When a user asks for a property estimate, try to gather required parameters if they are missing (e.g. rooms, distance, type, region), "
        "but if they provide a rough description, make reasonable assumptions for missing features (like 1 bathroom for 2 beds, distance=15, landsize=400, etc.) "
        "just so you can provide an estimate using predict_property_price, specifying what you assumed. "
        "Important Mapping for property_type: 'h' for house/villa, 'u' for unit/apartment, 't' for townhouse. "
        "Be polite, professional, and explain your reasoning clearly."
    ))

    # 3. Define Graph Nodes
    def chatbot_node(state: MessagesState):
        messages = state["messages"]
        # Inject system message safely
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [sys_msg] + messages
        response = invoke_with_pollinations_fallback(messages=messages, tools=tools)
        return {"messages": [response]}
        
    tool_node = ToolNode(tools=tools)
    
    # 4. Define Edge Logic
    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"
        
    # 5. Build StateGraph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", chatbot_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")
    
    # Compile Graph
    agent_app = workflow.compile()
    
    return agent_app
