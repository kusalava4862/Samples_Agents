"""
Proper ReAct Agent Implementation - Using Agent Reasoning for Medical Claim Validation
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, TypedDict
from langchain_core.tools import tool
from uipath_langchain.chat import UiPathAzureChatOpenAI
from uipath import UiPath
from pydantic import BaseModel
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from uipath.tracing import traced

# Configure logging for tracing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize
llm = UiPathAzureChatOpenAI(
    model="gpt-4o-2024-08-06",
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,
)

sdk = UiPath()

# Configuration
uipath_folder_asset_name = "UiPath_SDK_Challenge"
uipath_icd_codes_index_name = "ICD_Codes"
uipath_cpt_codes_index_name = "CPT_Codes"


class GraphState(TypedDict):
    """State for the validation graph with tracing support"""
    claim_data: str
    discharge_summary: str
    provider_email: Optional[str]
    agent_messages: List[Dict[str, Any]]
    validation_results: Dict[str, Any]


class GraphOutput(BaseModel):
    claim_valid: bool
    justification: str


# Tool 1: CPT Code Lookup
@tool
@traced()
async def cpt_lookup(code: str) -> str:
    """Lookup CPT code and description from index. Returns code and full description for agent reasoning.
    
    Args:
        code: CPT code to lookup (e.g., '99213')
    
    Returns:
        String containing CPT code and full description from the index
    """
    try:
        logger.info(f"Looking up CPT code: {code}")
        results = await sdk.context_grounding.search_async(
            name=uipath_cpt_codes_index_name,
            query=code,
            number_of_results=3,
            folder_path=uipath_folder_asset_name
        )
        
        if results and len(results) > 0:
            # Return the most relevant result with full description
            best_match = results[0]
            # Handle different response structures
            if hasattr(best_match, 'content'):
                content = best_match.content
            elif hasattr(best_match, 'text'):
                content = best_match.text
            elif hasattr(best_match, 'description'):
                content = best_match.description
            else:
                content = str(best_match)
            logger.info(f"Found CPT code {code} in index")
            return f"CPT Code: {code}\nDescription: {content}"
        else:
            logger.warning(f"CPT code {code} not found in index")
            return f"CPT Code: {code}\nDescription: NOT FOUND in database"
            
    except Exception as e:
        logger.error(f"Error looking up CPT code {code}: {str(e)}")
        return f"ERROR: Could not lookup CPT {code} - {str(e)}"


# Tool 2: ICD Code Lookup
@tool
@traced()
async def icd_lookup(code: str) -> str:
    """Lookup ICD code and description from index. Returns code and full description for agent reasoning.
    
    Args:
        code: ICD code to lookup (e.g., 'J020')
    
    Returns:
        String containing ICD code and full description from the index
    """
    try:
        logger.info(f"Looking up ICD code: {code}")
        results = await sdk.context_grounding.search_async(
            name=uipath_icd_codes_index_name,
            query=code,
            number_of_results=3,
            folder_path=uipath_folder_asset_name
        )
        
        if results and len(results) > 0:
            # Return the most relevant result with full description
            best_match = results[0]
            # Handle different response structures
            if hasattr(best_match, 'content'):
                content = best_match.content
            elif hasattr(best_match, 'text'):
                content = best_match.text
            elif hasattr(best_match, 'description'):
                content = best_match.description
            else:
                content = str(best_match)
            logger.info(f"Found ICD code {code} in index")
            return f"ICD Code: {code}\nDescription: {content}"
        else:
            logger.warning(f"ICD code {code} not found in index")
            return f"ICD Code: {code}\nDescription: NOT FOUND in database"
            
    except Exception as e:
        logger.error(f"Error looking up ICD code {code}: {str(e)}")
        return f"ERROR: Could not lookup ICD {code} - {str(e)}"


# Tool 3: Send Email
@tool
@traced()
async def send_email(email_data: str) -> str:
    """Send email notification via UiPath process. 
    
    Args:
        email_data: JSON string with keys: process_name, to, subject, html_body
    
    Returns:
        Status message indicating if email was sent successfully
    """
    try:
        data = json.loads(email_data)
        process_name = data.get('process_name', 'SendEmail')
        to_email = data.get('to')
        subject = data.get('subject', 'Claim Validation Notification')
        html_body = data.get('html_body', '')
        
        if not to_email:
            return "ERROR: 'to' email address is required"
        
        logger.info(f"Sending email to {to_email} via process {process_name}")
        
        job = await sdk.processes.invoke_async(
            name=process_name,
            input_arguments={
                'to': to_email,
                'subject': subject,
                'html_body': html_body
            },
            folder_path=uipath_folder_asset_name
        )
        
        logger.info(f"Email job created: {job.key}")
        return f"EMAIL_SENT: Notification sent to {to_email} via process {process_name}. Job ID: {job.key}"
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in email_data: {str(e)}")
        return f"ERROR: Invalid JSON format - {str(e)}"
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return f"ERROR: Could not send email - {str(e)}"


# System prompt for agent reasoning
SYSTEM_PROMPT = """You are a medical claim validation expert. Your task is to validate medical claims by reasoning about CPT and ICD codes.

AVAILABLE TOOLS:
- cpt_lookup(code): Lookup CPT code and description from database
- icd_lookup(code): Lookup ICD code and description from database  
- send_email(email_data): Send HTML email via UiPath process (JSON with: process_name, to, subject, html_body)

VALIDATION PROCESS - USE YOUR REASONING:
1. Extract all CPT and ICD codes from the claim data
2. For each code, use the lookup tools to retrieve the official description
3. COMPARE the retrieved description with the claim description - reason if they match
4. REASON about CPT-ICD medical alignment - are the procedures appropriate for the diagnoses?
5. VERIFY codes match the discharge summary - reason if the codes are supported by the clinical documentation
6. If claim is invalid, prepare a detailed HTML email with:
   - Claim ID and provider information
   - List of invalid codes with explanations
   - Specific reasons for each validation failure
   - Professional HTML formatting

CRITICAL REASONING GUIDELINES:
- A claim is valid ONLY if:
  * ALL codes exist in the database
  * Code descriptions match claim descriptions
  * CPT-ICD pairs are medically appropriate
  * Codes are supported by the discharge summary
- Use your medical knowledge to reason about alignment - don't rely on hardcoded rules
- When preparing HTML email, use proper HTML structure with tables, headers, and styling
- Always provide detailed justification for your decisions

Use the ReAct format: Thought → Action → Observation → Repeat until you have a final answer.
"""

@traced()
async def validate_cpt_icd_codes(state: GraphState) -> GraphOutput:
    """Main validation function using LangGraph ReAct agent with dynamic reasoning and tracing"""
    
    # Initialize tracing
    agent_messages = state.get('agent_messages', [])
    validation_results = state.get('validation_results', {})
    trace_spans = []
    
    try:
        logger.info("Starting claim validation with ReAct agent")
        trace_spans.append({
            "name": "validate_cpt_icd_codes",
            "status": "start",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Create tools list
        tools = [cpt_lookup, icd_lookup, send_email]
        
        # Create ReAct agent using LangGraph prebuilt
        from langchain_core.messages import SystemMessage
        
        agent_graph = create_react_agent(
            llm,
            tools
        )
        
        # Prepare input for the agent
        agent_input = f"""You need to validate the following medical claim:

CLAIM DATA:
{state['claim_data']}

DISCHARGE SUMMARY:
{state['discharge_summary']}

PROVIDER EMAIL: {state.get('provider_email', 'provider@example.com')}

TASK:
1. Extract all CPT and ICD codes from the claim data
2. Lookup each code to get official descriptions
3. Reason about code validity by comparing descriptions
4. Reason about CPT-ICD medical alignment
5. Reason if codes match the discharge summary
6. Determine if claim is valid
7. If invalid, prepare and send HTML email with detailed findings

Provide your final answer in this JSON format:
{{"claim_valid": true/false, "justification": "detailed reasoning"}}
"""
        
        # Initialize state for agent with system prompt
        from langchain_core.messages import SystemMessage, HumanMessage
        
        initial_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=agent_input)
        ]
        
        initial_state = {"messages": initial_messages}
        
        # create_react_agent returns a compiled graph
        app = agent_graph
        
        # Run the agent with streaming to capture dynamic nodes and edges
        config = {
            "configurable": {
                "thread_id": f"validation-{hash(state['claim_data'])}"
            }
        }
        logger.info("Invoking ReAct agent for validation with streaming")
        
        final_state = None
        executed_nodes = []
        executed_edges = []
        
        # Stream events to capture dynamic nodes and create traces
        async for event in app.astream(initial_state, config, stream_mode="updates"):
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
        final_state = await app.ainvoke(initial_state, config)
        
        # Store trace information in state (will be accessible via result)
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
        import os
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
                        "name": "validate_cpt_icd_codes",
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
                is_valid = "true" in output.lower() or "valid" in output.lower() and "invalid" not in output.lower()
                result = GraphOutput(
                    claim_valid=is_valid,
                    justification=output
                )
                
                # Complete trace
                trace_spans.append({
                    "name": "validate_cpt_icd_codes",
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
                "name": "validate_cpt_icd_codes",
                "status": "error",
                "timestamp": asyncio.get_event_loop().time(),
                "error": "Could not parse result"
            })
            
            return result
        else:
            logger.error("Agent did not return any messages")
            
            # Complete trace with error
            trace_spans.append({
                "name": "validate_cpt_icd_codes",
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
            "name": "validate_cpt_icd_codes",
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
builder.add_node("validate_cpt_icd_codes", validate_cpt_icd_codes)
builder.add_edge(START, "validate_cpt_icd_codes")
builder.add_edge("validate_cpt_icd_codes", END)

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
    """Test the proper ReAct agent"""
    
    initial_state: GraphState = {
        "claim_data": """{
    "claim_id": "CLM-2025-000201",
    "claim_lines": [
        {"line_id": "1", "cpt": "99213", "icd": "J020", "description": "Office visit"},
        {"line_id": "2", "cpt": "36415", "icd": "J020", "description": "Venipuncture"},
        {"line_id": "3", "cpt": "87880", "icd": "J020", "description": "Rapid strep test"}
    ]
}""",
        "discharge_summary": """Patient presented with severe sore throat and fever.
        Pharynx erythematous with white exudate on tonsils.
        Rapid strep test performed and returned positive.
        Diagnosis: Streptococcal pharyngitis""",
        "provider_email": "doctor@example.com",
        "agent_messages": [],
        "validation_results": {}
    }
    
    print("\n" + "="*60)
    print("RUNNING PROPER REACT AGENT VALIDATION")
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
    
    # Get final result - note: GraphOutput only returns claim_valid and justification
    # Trace info is in validation_results which is stored in state
    result = await graph.ainvoke(initial_state)
    
    # Get trace info from final state if available
    import os
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