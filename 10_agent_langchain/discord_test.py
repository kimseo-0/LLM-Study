# openai 토큰을 .env 추가 -> OPENAI_API_KEY
# 디스코드 봇 토큰을 .env 추가 -> DISCORD_BOT_TOKEN
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. 모델 설정
llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature = 0,
)

# 2. 도구 생성
from langchain_discord import DiscordWebhookTool
from langchain_discord_shikenso import DiscordToolkit

discord_webhook = DiscordWebhookTool()
discord_tools = DiscordToolkit().get_tools() + [discord_webhook]

# 3. 프롬프트 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 디스코드를 관리하는 AI 비서야. 주어지는 도구를 잘 활용해서 사용자의 요청에 답하도록 해"),
    ("user", "{question}"),
    ("placeholder", "{agent_scratchpad}")
])

# 4. 단일 에이전트 생성
agent = create_openai_tools_agent(
    llm=llm,
    tools=discord_tools,
    prompt=prompt
)

# 5. excutor 설정
executor = AgentExecutor(
    agent = agent,
    tools=discord_tools,
    verbose=True
)

channel_id = 1417295078568886326

# executor.invoke({'question' : "디스코드 {channel_id} 채널의 메세지를 5개 읽어와줘"})
# executor.invoke({'question' : f"디스코드 {channel_id} 채널에 메세지를 HELLO라고 보내주는데 빨강 컬러로 카드형태로 보내줘"})
executor.invoke({'question' : f"디스코드 웹훅으로 HELLO라고 보내줘"})