import os
import joblib
import pandas as pd
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

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

# Define the tools array
tools = [predict_property_price, get_market_statistics]

# ---------------------------------------------------------------------------
# LangGraph Workflow Definition
# ---------------------------------------------------------------------------
def build_langgraph_agent(api_key: str):
    """
    Constructs and returns the LangGraph application.
    """
    # 1. Initialize LLM with tools
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. System prompt
    sys_msg = SystemMessage(content=(
        "You are 'Melbourne EstateAI', an expert real estate agent and predictive assistant. "
        "You can answer user questions about Melbourne real estate and perform highly accurate price predictions using your tools. "
        "When a user asks for a property estimate, try to gather required parameters if they are missing (e.g. rooms, distance, type, region), "
        "but if they provide a rough description, make reasonable assumptions for missing features (like 1 bathroom for 2 beds, distance=15, landsize=400, etc.) "
        "just so you can provide an estimate using the predict_property_price tool, then specify what you assumed. "
        "Important Mapping for property_type: 'h' for house/villa, 'u' for unit/apartment, 't' for townhouse. "
        "Be polite, professional, and explain your reasoning clearly."
    ))

    # 3. Define Graph Nodes
    def chatbot_node(state: MessagesState):
        messages = state["messages"]
        # Inject system message safely
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [sys_msg] + messages
        response = llm_with_tools.invoke(messages)
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
