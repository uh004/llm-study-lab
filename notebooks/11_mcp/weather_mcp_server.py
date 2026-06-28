import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("Weather_MCP_Server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="주어진 도시의 날씨를 알려줍니다. (예: Seoul, Tokyo)",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_weather":
        city = arguments.get("city")
        fake_db = {"Seoul": "맑음, 25도", "Tokyo": "비, 20도", "New York": "눈, -2도"}
        result = fake_db.get(city, "해당 도시의 날씨를 모릅니다.")
        return [TextContent(type="text", text=result)]
    raise ValueError(f"알 수 없는 도구입니다: {name}")

async def main():
    # 터미널 표준 입출력(stdio)을 통해 MCP 서버를 오픈합니다.
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
