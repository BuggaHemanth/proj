import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from typing import Any
import uuid

# Set page config
st.set_page_config(page_title="Travel Agent Chat", page_icon="✈️", layout="wide")


@st.cache_resource
def create_travel_graph():
    """Create and cache the travel agent graph"""
    
    # Create model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    class MultiAgentState(MessagesState):
        last_active_agent: str

    def get_travel_recommendations(query: str) -> str:
        """Get travel destination recommendations based on query."""
        if "beach" in query.lower():
            return "For a beach vacation, I recommend Hawaii, Bali, or the Maldives."
        elif "mountain" in query.lower():
            return "For a mountain getaway, consider the Swiss Alps, the Rockies, or the Himalayas."
        elif "city" in query.lower():
            return "For a city break, I suggest exploring New York, Paris, or Tokyo."
        else:
            return "I recommend visiting popular destinations like Hawaii, Bali, or Paris."

    def get_hotel_recommendations(query: str) -> str:
        """Get hotel recommendations based on query."""
        if "beach" in query.lower():
            return "For beach hotels, I recommend The Ritz-Carlton, Bali or Four Seasons Maui."
        elif "city" in query.lower():
            return "For city hotels, consider The Peninsula, Hong Kong or The Savoy, London."
        else:
            return "I recommend checking out hotels like The Ritz-Carlton or Four Seasons."

    def make_handoff_tool(agent_name: str) -> BaseTool:
        class HandoffTool(BaseTool):
            name: str = f"handoff_to_{agent_name}"
            description: str = f"Hand off the conversation to {agent_name}."
            
            def _run(self, tool_input: str, **kwargs: Any) -> str:
                return f"Switching to {agent_name}"
            
            async def _arun(self, tool_input: str, **kwargs: Any) -> str:
                return f"Switching to {agent_name}"
        
        return HandoffTool()

    # Create agents
    travel_advisor_tools = [
        get_travel_recommendations,
        make_handoff_tool(agent_name="hotel_advisor"),
    ]

    travel_advisor = create_react_agent(
        model,
        travel_advisor_tools,
        prompt=(
            "You are a general travel expert that can recommend travel destinations "
            "(e.g. countries, cities, etc). If you need hotel recommendations, ask 'hotel_advisor' for help. "
            "You MUST include a human-readable response before transferring to another agent."
        ),
    )

    hotel_advisor_tools = [
        get_hotel_recommendations,
        make_handoff_tool(agent_name="travel_advisor"),
    ]

    hotel_advisor = create_react_agent(
        model,
        hotel_advisor_tools,
        prompt=(
            "You are a hotel expert that can provide hotel recommendations for a given destination. "
            "If you need help picking travel destinations, ask 'travel_advisor' for help. "
            "You MUST include a human-readable response before transferring to another agent."
        ),
    )

    def call_travel_advisor(state: MultiAgentState) -> Command:
        response = travel_advisor.invoke(state)
        update = {**response, "last_active_agent": "travel_advisor"}
        return Command(update=update, goto="human")

    def call_hotel_advisor(state: MultiAgentState) -> Command:
        response = hotel_advisor.invoke(state)
        update = {**response, "last_active_agent": "hotel_advisor"}
        return Command(update=update, goto="human")

    # Build the graph
    builder = StateGraph(MultiAgentState)
    builder.add_node("travel_advisor", call_travel_advisor)
    builder.add_node("hotel_advisor", call_hotel_advisor)
    builder.add_edge(START, "travel_advisor")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph, MultiAgentState

# Initialize the app
st.title("✈️ Travel Agent Multi-Agent System")
st.markdown("Chat with our AI travel advisors to plan your perfect trip!")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about travel destinations or hotels..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get graph and state
        graph, MultiAgentState = create_travel_graph()
        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Process with agents - SHOW REAL-TIME INTERACTIONS
        with st.chat_message("assistant"):
            inputs = {"messages": [{"role": "user", "content": prompt}]}
            
            # Create containers for real-time updates
            status_container = st.empty()
            interaction_container = st.container()
            
            agent_responses = []
            step_count = 0
            
            try:
                with status_container.container():
                    st.info("🤖 **Agent Workflow Started**")
                
                for update in graph.stream(inputs, config=thread_config, stream_mode="updates"):
                    for node_id, value in update.items():
                        step_count += 1
                        
                        with interaction_container:
                            # Show which agent is active
                            if node_id in ["travel_advisor", "hotel_advisor"]:
                                agent_emoji = "🏨" if "hotel" in node_id else "✈️"
                                st.write(f"**Step {step_count}:** {agent_emoji} **{node_id.title()}** is thinking...")
                                
                                # Show the agent's tools being used
                                if isinstance(value, dict):
                                    if "messages" in value and value["messages"]:
                                        last_message = value["messages"][-1]
                                        if hasattr(last_message, 'content'):
                                            # Display agent response with nice formatting
                                            with st.expander(f"{agent_emoji} {node_id.title()} Response", expanded=True):
                                                st.write(last_message.content)
                                            
                                            response_text = f"{agent_emoji} **{node_id.title()}**: {last_message.content}"
                                            agent_responses.append(response_text)
                                    
                                    # Show state updates
                                    if "last_active_agent" in value:
                                        st.caption(f"🔄 **Active Agent:** {value['last_active_agent']}")
                            
                            elif node_id == "human":
                                st.write(f"**Step {step_count}:** 👤 **Human Input** processed")
                        
                        # Add small delay to show the flow
                        import time
                        time.sleep(0.5)
                
                # Final status
                with status_container.container():
                    st.success(f"✅ **Workflow Complete** - {step_count} steps executed")
                
                # Add summary to chat history
                if agent_responses:
                    full_response = "\n\n".join(agent_responses)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                with status_container.container():
                    st.error(f"❌ **Error**: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})

with col2:
    st.subheader("🔧 Agent Workflow Monitor")
    
    # Real-time workflow status
    if "workflow_status" not in st.session_state:
        st.session_state.workflow_status = "Ready"
    
    status_placeholder = st.empty()
    with status_placeholder.container():
        st.metric("Current Status", st.session_state.workflow_status)
    
    # Show agent information
    st.markdown("""
    **🤖 Active Agents:**
    
    ✈️ **Travel Advisor**
    - Destination recommendations
    - Travel planning expertise
    - Can handoff to Hotel Advisor
    
    🏨 **Hotel Advisor** 
    - Hotel suggestions
    - Accommodation expertise
    - Can handoff to Travel Advisor
    
    **🔄 Workflow Features:**
    - ✅ Real-time step visualization
    - ✅ Agent handoff tracking
    - ✅ State monitoring
    - ✅ Interactive responses
    """)
    
    # Agent interaction log
    st.subheader("📊 Interaction Log")
    if "interaction_log" not in st.session_state:
        st.session_state.interaction_log = []
    
    if st.session_state.interaction_log:
        for i, log_entry in enumerate(st.session_state.interaction_log[-5:]):  # Show last 5
            st.text(f"{i+1}. {log_entry}")
    else:
        st.text("No interactions yet...")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interaction_log = []
        st.session_state.workflow_status = "Ready"
        st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Ask about destinations first, then request hotel recommendations for a complete travel plan!")