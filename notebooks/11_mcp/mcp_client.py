import os
import sys
import asyncio
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

# 환경 변수 로드
load_dotenv()

# 사용할 LLM 세팅
llm = ChatOpenAI(model="gpt-4o-mini")

# 서버 실행 파라미터 (weather_mcp_server.py)
python_exe = sys.executable 
server_params = StdioServerParameters(
    command=python_exe,
    args=["weather_mcp_server.py"],
)

async def run_mcp_agent():
    async with AsyncExitStack() as stack:
        print("🔌 [1단계] MCP 서버와 stdio 통신 파이프라인을 엽니다...")
        # 터미널 환경에서는 sys.stderr를 기본으로 사용해도 충돌이 안 납니다.
        read, write = await stack.enter_async_context(stdio_client(server_params))
        
        print("🔗 [2단계] 클라이언트 세션 연결 중...")
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        
        print("🛠️ [3단계] 서버에 등록된 툴(도구)들을 LangChain 규격으로 가져옵니다...")
        tools = await load_mcp_tools(session)
        print(f"✅ 가져온 도구 목록: {[t.name for t in tools]}")

        # 에이전트 생성
        agent = create_agent(
            model=llm, 
            tools=tools, 
            system_prompt="당신은 유능한 비서입니다. 주어진 도구를 사용하여 날씨를 확인하세요."
        )

        # 사용자 질문 루프
        print("\n================================================")
        print("🤖 MCP 에이전트가 준비되었습니다! (종료하려면 'q' 입력)")
        print("================================================\n")
        
        while True:
            question = input("사용자 질문: ")
            if question.lower() == 'q':
                print("종료합니다.")
                break
                
            result = await agent.ainvoke({"messages": [("user", question)]})
            print(f"\n[에이전트 답변]:\n{result['messages'][-1].content}\n")

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
