from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
            "CustomWeather",
            host="0.0.0.0",
            port=8100
            )

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location"""
    return f"{location}의 날씨는 맑음입니다"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")