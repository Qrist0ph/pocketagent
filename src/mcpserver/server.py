# math_server.py
#https://github.com/langchain-ai/langchain-mcp-adapters

import logging
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


# Configure logging to a file
logging.basicConfig(filename="/tmp/math_server.log", level=logging.INFO, format="%(asctime)s %(message)s")



@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    msg = f"Adding {a} + {b}"
    logging.info(msg)
    with open("/tmp/foo", "a") as f:
        f.write(msg + "\n")
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    msg = f"Multiplying {a} * {b}"
    logging.info(msg)
    with open("/tmp/foo", "a") as f:
        f.write(msg + "\n")
    return a * b

if __name__ == "__main__":
    logging.info("Starting Math MCP server...")
    # Create /tmp/foo
    with open("/tmp/foo", "a") as f:
        f.write("created2 by math_server.py\n")
    mcp.run(transport="stdio")