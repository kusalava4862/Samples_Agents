"""
Test script for MCP server connection and tool loading
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
mcp_server_url = os.getenv("UIPATH_MCP_SERVER_URL", "https://staging.uipath.com/skillrpa/DefaultTenant/agenthub_/mcp/428a6c1b-8772-4ebb-82ae-4df110d2a2a0/testserver")
uipath_access_token = os.getenv("UIPATH_ACCESS_TOKEN") or os.getenv("UIPATH_TOKEN")


@asynccontextmanager
async def get_mcp_session():
    """MCP session management"""
    if not mcp_server_url:
        raise ValueError("UIPATH_MCP_SERVER_URL environment variable is required")
    
    if not uipath_access_token:
        logger.warning("UIPATH_ACCESS_TOKEN not found in .env file. Connection may fail if authentication is required.")
        headers = {}
    else:
        headers = {"Authorization": f"Bearer {uipath_access_token}"}
        logger.info("Using access token from .env file for MCP connection")
    
    async with streamablehttp_client(
        url=mcp_server_url,
        headers=headers if headers else None,
        timeout=60,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def get_mcp_tools():
    """Load MCP tools for use with agents"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        logger.info(f"Loaded {len(tools)} tools from MCP server")
        return tools


async def main():
    """Main test function"""
    print("=" * 60)
    print("Testing MCP Server Connection")
    print("=" * 60)
    print(f"MCP Server URL: {mcp_server_url}")
    print(f"Access Token: {'***' + uipath_access_token[-4:] if uipath_access_token else 'Not provided'}")
    print("=" * 60)
    print()
    
    try:
        tools = await get_mcp_tools()
        
        print(f"\n✓ Successfully connected to MCP server")
        print(f"✓ Loaded {len(tools)} tools\n")
        
        print("Available tools:")
        print("-" * 60)
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                print(f"   Description: {tool.description}")
            print()
        
        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
        return tools
        
    except Exception as e:
        logger.error(f"Error connecting to MCP server: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    asyncio.run(main())