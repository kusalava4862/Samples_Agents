"""
Script to inspect the graph structure programmatically
Run this to see the graph nodes and edges
"""

try:
    from validate_cpt_icd_codes import graph
    
    print("=" * 60)
    print("OUTER GRAPH STRUCTURE")
    print("=" * 60)
    
    # Get graph structure
    if hasattr(graph, 'nodes'):
        print(f"\nNodes: {list(graph.nodes.keys())}")
    
    if hasattr(graph, 'edges'):
        print(f"\nEdges: {list(graph.edges)}")
    
    # Get the compiled graph structure
    if hasattr(graph, 'get_graph'):
        inner_graph = graph.get_graph()
        print(f"\nInner graph type: {type(inner_graph)}")
        if hasattr(inner_graph, 'nodes'):
            print(f"Inner graph nodes: {list(inner_graph.nodes.keys())}")
    
    print("\n" + "=" * 60)
    print("GRAPH FLOW")
    print("=" * 60)
    print("""
    START
      ↓
    validate_cpt_icd_codes (Node)
      ↓
        [Contains ReAct Agent]
          ↓
        ReAct Agent Loop:
          - Action (LLM decides tool)
          - Tool Execution (cpt_lookup | icd_lookup | send_email)
          - Observation
          - Loop back or Finish
      ↓
    END
    """)
    
    print("\n" + "=" * 60)
    print("REACT AGENT STRUCTURE")
    print("=" * 60)
    print("""
    The ReAct agent is created by create_react_agent() and contains:
    
    1. Agent Start
    2. Should Continue? (conditional node)
    3. Action Node (LLM generates tool call)
    4. Tool Execution (one of 3 tools)
    5. Observation (tool results)
    6. Loop back to "Should Continue?"
    7. Final Answer (when done)
    
    The agent dynamically decides:
    - Which tool to call next
    - When to stop reasoning
    - What the final answer is
    """)
    
except ImportError as e:
    print(f"Could not import graph: {e}")
    print("\nMake sure all dependencies are installed:")
    print("  pip install uipath-langchain langchain langgraph")

