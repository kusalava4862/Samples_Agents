"""
Policy Terms and Limits Validation Agent - Using MCP Server Tools
Validates medical claims against policy terms, coverage, exclusions, and usage limits
"""

import json
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List, TypedDict
from contextlib import asynccontextmanager
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from uipath_langchain.chat import UiPathAzureChatOpenAI
from uipath import UiPath
from uipath.tracing import traced
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Configure logging for tracing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = UiPathAzureChatOpenAI(
    model="gpt-4o-2024-08-06",
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,
)

# Initialize SDK for authentication
sdk = UiPath()

# Configuration
uipath_folder_asset_name = "UiPath_SDK_Challenge"
uipath_mcp_server_url = os.getenv("UIPATH_MCP_SERVER_URL")


class GraphState(TypedDict):
    """State for the policy validation graph with tracing support"""
    claim_data: str
    agent_messages: List[Dict[str, Any]]
    validation_results: Dict[str, Any]


class GraphOutput(BaseModel):
    claim_valid: bool
    justification: str


@asynccontextmanager
async def get_mcp_tools():
    """Async context manager to get MCP tools from UiPath MCP server"""
    if not uipath_mcp_server_url:
        raise ValueError("UIPATH_MCP_SERVER_URL environment variable is required")
    
    # SDK authentication is handled via environment variables
    # MCP server should use the same authentication mechanism
    # Try to get token from environment variables that SDK uses
    # Common UiPath env vars: UIPATH_CLIENT_ID, UIPATH_CLIENT_SECRET, UIPATH_REFRESH_TOKEN, etc.
    # For MCP, we may need to pass the token if available
    auth_token = os.getenv("UIPATH_ACCESS_TOKEN") or os.getenv("UIPATH_TOKEN")
    
    # Prepare headers for MCP connection
    headers = None
    if auth_token:
        headers = {"Authorization": f"Bearer {auth_token}"}
        logger.info("Using access token from environment variable for MCP connection")
    else:
        # SDK uses environment variables for authentication
        # MCP server should leverage the same authentication context
        # If no explicit token, the MCP server may use SDK's authentication mechanism
        logger.info("Using SDK authentication context for MCP connection")
    
    async with streamablehttp_client(
        url=uipath_mcp_server_url,
        headers=headers,
        timeout=60,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(tools)} tools from MCP server: {[tool.name for tool in tools]}")
            yield tools


# System prompt for agent reasoning
SYSTEM_PROMPT = """You are a medical claim policy validation expert. Your task is to validate medical claims against policy terms, coverage, exclusions, and usage limits.

AVAILABLE MCP TOOLS:
1. get_policy_details(policy_id: str)
   - Input: policy_id (string)
   - Output: Returns policy details including deductible, effective_date, expiry_date, max_outpocket, plan_name, status
   - Use this to check if policy is active on the date of service

2. get_coverage_for_cpt_code(cpt_code: str)
   - Input: cpt_code (comma-separated string, e.g., "99213,36415,87880")
   - Output: Returns array of coverage information for each CPT code
   - Format: "{cpt code - <code>, coverage limit - <limit>, frequency limit - <limit>}"
   - Use this to check if CPT codes are covered and get coverage/frequency limits

3. get_exclusion_of_code(cpt_code: str, icd_code: str, patient_id: str)
   - Input: cpt_code (comma-separated string), icd_code (comma-separated string), patient_id (string)
   - Output: Returns array of exclusion details
   - Format: "{codetype - cpt/icd, code - <code>, reson - <reason>}"
   - Use this to check if any CPT/ICD combinations are excluded
   - If exclusion data is not available, the array will be empty

VALIDATION PROCESS - USE YOUR REASONING:
1. Extract from claim_data JSON:
   - patient_id
   - policy_id
   - date_of_service (or service_date)
   - All CPT codes from claim_lines (create comma-separated string)
   - All ICD codes from claim_lines (create comma-separated string)

2. Call get_policy_details(policy_id) to get policy information
   - Check if policy status is "active"
   - Verify date_of_service is within effective_date and expiry_date range
   - Policy must be active on the date of service

3. Call get_coverage_for_cpt_code(cpt_codes) with comma-separated CPT codes
   - Check if all CPT codes are covered (coverage array should have entries for all codes)
   - Review coverage limits and frequency limits for each CPT code
   - Verify claim amounts and frequencies are within limits

4. Call get_exclusion_of_code(cpt_codes, icd_codes, patient_id) with comma-separated codes
   - Check if any exclusions are returned
   - If exclusion_details array has entries, those combinations are excluded
   - If exclusion data is not available (empty array), assume no exclusions

5. Determine if claim is valid based on all checks

CRITICAL VALIDATION RULES:
A claim is valid ONLY if ALL of the following are true:
- Policy is active (status = "active") and date_of_service is within effective_date to expiry_date
- All CPT codes in the claim are covered (coverage array has entries for all codes)
- No CPT/ICD combinations are excluded (exclusion_details array is empty or not available)
- All services are within coverage limits and frequency limits

If any condition fails, the claim is invalid. Provide detailed justification explaining which validation failed and why.

Use the ReAct format: Thought → Action → Observation → Repeat until you have a final answer.

Provide your final answer in this JSON format:
{"claim_valid": true/false, "justification": "detailed reasoning"}
"""


@traced()
async def validate_policy_terms(state: GraphState) -> GraphOutput:
    """Main validation function using LangGraph ReAct agent with MCP server tools"""
    
    # Initialize tracing
    agent_messages = state.get('agent_messages', [])
    validation_results = state.get('validation_results', {})
    trace_spans = []
    
    try:
        logger.info("Starting policy terms validation with ReAct agent")
        trace_spans.append({
            "name": "validate_policy_terms",
            "status": "start",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Get MCP tools
        async with get_mcp_tools() as tools:
            logger.info(f"Using {len(tools)} MCP tools for validation")
            
            # Create ReAct agent using LangGraph prebuilt
            agent_graph = create_react_agent(llm, tools)
            
            # Prepare input for the agent
            agent_input = f"""You need to validate the following medical claim against policy terms and limits:

CLAIM DATA:
{state['claim_data']}

TASK:
1. Extract patient_id, policy_id, date_of_service, and all CPT/ICD codes from the claim_data JSON
2. Create comma-separated strings for CPT codes and ICD codes
3. Call get_policy_details to check if policy is active on date of service
4. Call get_coverage_for_cpt_code to check coverage and limits for all CPT codes
5. Call get_exclusion_of_code to check for any exclusions
6. Validate all requirements and determine if claim is valid
7. Provide detailed justification for your decision

Provide your final answer in this JSON format:
{{"claim_valid": true/false, "justification": "detailed reasoning"}}
"""
            
            # Initialize state for agent with system prompt
            initial_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=agent_input)
            ]
            
            initial_state = {"messages": initial_messages}
            
            # Run the agent with streaming to capture dynamic nodes and edges
            config = {
                "configurable": {
                    "thread_id": f"policy-validation-{hash(state['claim_data'])}"
                }
            }
            logger.info("Invoking ReAct agent for policy validation with streaming")
            
            executed_nodes = []
            executed_edges = []
            
            # Stream events to capture dynamic nodes and create traces
            async for event in agent_graph.astream(initial_state, config, stream_mode="updates"):
                # Track each node execution
                for node_name, node_data in event.items():
                    if node_name not in executed_nodes:
                        executed_nodes.append(node_name)
                        logger.info(f"Executing node: {node_name}")
                        trace_spans.append({
                            "name": node_name,
                            "type": "node",
                            "status": "start",
                            "timestamp": asyncio.get_event_loop().time()
                        })
                        
                        # Track edges based on node execution order
                        if len(executed_nodes) > 1:
                            edge = (executed_nodes[-2], node_name)
                            if edge not in executed_edges:
                                executed_edges.append(edge)
                                logger.info(f"Executing edge: {edge[0]} -> {edge[1]}")
                        
                        # Extract message content for tracing
                        if isinstance(node_data, dict) and "messages" in node_data:
                            messages = node_data["messages"]
                            if messages:
                                last_msg = messages[-1]
                                msg_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                                agent_messages.append({
                                    "node": node_name,
                                    "message": msg_content[:200] if len(str(msg_content)) > 200 else str(msg_content),
                                    "timestamp": asyncio.get_event_loop().time()
                                })
                                logger.debug(f"Node {node_name} message: {msg_content[:100]}...")
                        
                        # Mark node as completed
                        trace_spans.append({
                            "name": node_name,
                            "type": "node",
                            "status": "end",
                            "timestamp": asyncio.get_event_loop().time()
                        })
            
            # Get final state
            final_state = await agent_graph.ainvoke(initial_state, config)
            
            # Store trace information in state
            validation_results = {
                "executed_nodes": executed_nodes,
                "executed_edges": executed_edges,
                "trace_spans": trace_spans,
                "total_nodes": len(executed_nodes),
                "total_edges": len(executed_edges)
            }
            
            # Update state with trace information
            state['validation_results'] = validation_results
            state['agent_messages'] = agent_messages
            
            logger.info(f"Agent execution completed. Nodes: {executed_nodes}, Edges: {executed_edges}")
            
            # Write trace file in JSON Lines format (UiPath SDK pattern)
            trace_file = os.environ.get('TRACE_FILE', 'traces.jsonl')
            try:
                with open(trace_file, 'a') as f:
                    for span in trace_spans:
                        f.write(json.dumps(span) + '\n')
                logger.info(f"Traces written to {trace_file}")
            except Exception as trace_error:
                logger.warning(f"Could not write trace file: {trace_error}")
            
            # Extract final answer from agent messages
            if final_state and final_state.get("messages"):
                last_message = final_state["messages"][-1]
                output = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                logger.info(f"Agent completed. Extracting final answer from: {output[:200]}...")
                
                # Try to parse JSON from output
                import re
                json_match = re.search(r'\{[^{}]*"claim_valid"[^{}]*\}', output, re.DOTALL)
                
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        result = GraphOutput(
                            claim_valid=parsed.get("claim_valid", False),
                            justification=parsed.get("justification", output)
                        )
                        logger.info(f"Validation complete: Claim valid={result.claim_valid}")
                        
                        # Complete trace
                        trace_spans.append({
                            "name": "validate_policy_terms",
                            "status": "end",
                            "timestamp": asyncio.get_event_loop().time(),
                            "result": {"claim_valid": result.claim_valid}
                        })
                        
                        return result
                    except json.JSONDecodeError:
                        logger.warning("Could not parse JSON, using full output as justification")
                
                # Fallback: extract from text
                if "claim_valid" in output.lower() or "valid" in output.lower():
                    # Try to infer from text
                    is_valid = "true" in output.lower() or ("valid" in output.lower() and "invalid" not in output.lower())
                    result = GraphOutput(
                        claim_valid=is_valid,
                        justification=output
                    )
                    
                    # Complete trace
                    trace_spans.append({
                        "name": "validate_policy_terms",
                        "status": "end",
                        "timestamp": asyncio.get_event_loop().time(),
                        "result": {"claim_valid": result.claim_valid}
                    })
                    
                    return result
                
                result = GraphOutput(
                    claim_valid=False,
                    justification=f"Could not parse result. Agent output: {output}"
                )
                
                # Complete trace with error
                trace_spans.append({
                    "name": "validate_policy_terms",
                    "status": "error",
                    "timestamp": asyncio.get_event_loop().time(),
                    "error": "Could not parse result"
                })
                
                return result
            else:
                logger.error("Agent did not return any messages")
                
                # Complete trace with error
                trace_spans.append({
                    "name": "validate_policy_terms",
                    "status": "error",
                    "timestamp": asyncio.get_event_loop().time(),
                    "error": "Agent did not return any messages"
                })
                
                return GraphOutput(
                    claim_valid=False,
                    justification="Agent did not return any validation result"
                )
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}", exc_info=True)
        
        # Complete trace with exception
        trace_spans.append({
            "name": "validate_policy_terms",
            "status": "error",
            "timestamp": asyncio.get_event_loop().time(),
            "error": str(e),
            "exception_type": type(e).__name__
        })
        
        return GraphOutput(
            claim_valid=False,
            justification=f"Error during validation: {str(e)}"
        )


# Build the graph with dynamic structure
# The ReAct agent handles its own internal flow with dynamic nodes/edges
builder = StateGraph(GraphState, output_schema=GraphOutput)
builder.add_node("validate_policy_terms", validate_policy_terms)
builder.add_edge(START, "validate_policy_terms")
builder.add_edge("validate_policy_terms", END)

# Compile graph with tracing support
graph = builder.compile()


# Function to get dynamic graph structure from execution
def get_dynamic_graph_structure(state: GraphState) -> Dict[str, Any]:
    """Extract dynamic graph structure from state"""
    validation_results = state.get('validation_results', {})
    return {
        "nodes": validation_results.get('executed_nodes', []),
        "edges": validation_results.get('executed_edges', []),
        "trace_spans": validation_results.get('trace_spans', []),
        "total_nodes": validation_results.get('total_nodes', 0),
        "total_edges": validation_results.get('total_edges', 0)
    }


# Test function
async def main():
    """Test the policy validation agent"""
    
    initial_state: GraphState = {
        "claim_data": """{
    "claim_id": "CLM-2025-000201",
    "patient_id": "PAT-001",
    "policy_id": "POL-12345",
    "date_of_service": "2025-01-15",
    "claim_lines": [
        {"line_id": "1", "cpt": "99213", "icd": "J020", "description": "Office visit", "amount": 150.00},
        {"line_id": "2", "cpt": "36415", "icd": "J020", "description": "Venipuncture", "amount": 25.00},
        {"line_id": "3", "cpt": "87880", "icd": "J020", "description": "Rapid strep test", "amount": 45.00}
    ]
}""",
        "agent_messages": [],
        "validation_results": {}
    }
    
    print("\n" + "="*60)
    print("RUNNING POLICY TERMS VALIDATION AGENT")
    print("="*60)
    print("\nWatch the agent's reasoning process below...")
    print("="*60 + "\n")
    
    # Run with streaming to see dynamic nodes and traces
    print("\nExecuting graph with tracing...")
    print("-" * 60)
    
    # Stream to see nodes as they execute and capture trace info
    final_state_dict = {}
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_data in event.items():
            print(f"✓ Node executed: {node_name}")
            final_state_dict.update(node_data)
            if isinstance(node_data, dict):
                # Show state updates if available
                if 'validation_results' in node_data:
                    results = node_data['validation_results']
                    if 'executed_nodes' in results:
                        print(f"  → Dynamic nodes: {results['executed_nodes']}")
                    if 'executed_edges' in results:
                        print(f"  → Dynamic edges: {results['executed_edges']}")
                    if 'trace_spans' in results:
                        print(f"  → Trace spans: {len(results['trace_spans'])}")
    
    # Get final result
    result = await graph.ainvoke(initial_state)
    
    # Get trace info from final state if available
    trace_info = final_state_dict.get('validation_results', {})
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Claim Valid: {result['claim_valid']}")
    print(f"Justification: {result['justification']}")
    
    # Show dynamic graph structure from trace info
    if trace_info:
        print("\n" + "="*60)
        print("DYNAMIC GRAPH STRUCTURE")
        print("="*60)
        print(f"Executed Nodes: {trace_info.get('executed_nodes', [])}")
        print(f"Executed Edges: {trace_info.get('executed_edges', [])}")
        print(f"Total Nodes: {trace_info.get('total_nodes', 0)}")
        print(f"Total Edges: {trace_info.get('total_edges', 0)}")
        print(f"Trace Spans: {len(trace_info.get('trace_spans', []))}")
        print(f"\nTrace file: {os.environ.get('TRACE_FILE', 'traces.jsonl')}")
    
    print("="*60)
    
    return result


if __name__ == "__main__":
    asyncio.run(main())

