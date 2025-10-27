import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()
SMITHERY_AI_KEY = os.getenv('SMITHERY_AI_KEY')
SMITHERY_AI_PROFILE = os.getenv('SMITHERY_AI_PROFILE')


from urllib.parse import urlencode
params = urlencode({"api_key": SMITHERY_AI_KEY, "profile": SMITHERY_AI_PROFILE})


# 실제 실행 파트
async def main():
    print("멀티 클라이언트 세팅 중...")
    
    client = MultiServerMCPClient(
        {
            # "Math" : {
            #     "command" : "python",
            #     "args" : ["C:/Potenup/LLM-Study/12_mcp/odd_math_server.py"],
            #     "transport" : "stdio"
            # },
            # "CustomWeather" : {
            #     "url" : "http://localhost:8100/mcp",
            #     "transport" : "streamable_http"
            # },
            # "mcp_weather_server": {
            #     "url": f"https://server.smithery.ai/@isdaniel/mcp_weather_server/mcp?{params}",
            #     "transport" : "streamable_http"
            # },
            "mcp_weather_server": {
                "transport" : "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@isdaniel/mcp_weather_server",
                    "--key",
                    SMITHERY_AI_KEY,
                    "--profile",
                    SMITHERY_AI_PROFILE
                ]
            }
        }
    )

    tools = await client.get_tools()
    for tool in tools:
        print(tool.name)
    
    print("=" * 30)

    model = ChatOpenAI(
        model = "gpt-4.1-mini",
        temperature=0
    )
    
    agent_excutor = create_react_agent(model=model, tools=tools)

    # # 더하기 도구
    # system_message = SystemMessage(
    #     content="너는 주어진 도구를 사용해서 잘못된 수학 계산을 하는 MCP 에이전트야"
    # )
    # response = await agent_excutor.ainvoke(
    #     {"messages" : [system_message, HumanMessage(content="1+2 는?")]}
    # )
    # print("답변: ", response["messages"][-1].content)

    # # 곱하기 도구
    # response = await agent_excutor.ainvoke(
    #     {"messages" : [system_message, HumanMessage(content="2*3 는?")]}
    # )
    # print("답변: ", response["messages"][-1].content)

    # 날씨 도구
    system_message = SystemMessage(
        content="너는 주어진 도구를 사용해서 날씨를 알려주는 에이전트야"
    )
    response = await agent_excutor.ainvoke(
        {"messages" : [HumanMessage(content="석촌동의 날씨를 알려줘")]}
    )
    print("답변: ", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())