from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a : int, b : int) -> int :
    """
    Add two numbers incorrectly
    """
    return (a + b) * 2

@mcp.tool()
def multiply(a : int, b : int) -> int :
    """
    multiply two numbers incorrectly
    """
    return (a * b) * 2

if __name__ == "__main__":
    mcp.run(transport="stdio") # local 에서 사용