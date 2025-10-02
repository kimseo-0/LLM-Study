from langchain_openai import ChatOpenAI
from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from fastapi import WebSocket 

# 환경 변수 불러오기
from dotenv import load_dotenv
load_dotenv()

# model
def create_model(model_name = "gpt-4.1-mini"):
    model = ChatOpenAI(
        temperature=0.1, # 창의력 정도
        model = model_name,
        verbose=True
    )
    return model

# prompt
def create_prompt():
    # 1. 프롬프트에 history 자리 확보
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 AI 도우미야, 간략하게 그냥 응답하도록 해"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}"),
    ])

    return prompt

# chain
def create_chain(prompt, model):
    chain = prompt | model | StrOutputParser()

    return chain

# history 
# 2. 대화 내용 저장소 만들기
stores : Dict[str, InMemoryChatMessageHistory] = {}
def get_store(session_id: str):
    print(f"[대화 세션ID]: {session_id}")
    if session_id not in stores:
        stores[session_id] = InMemoryChatMessageHistory()
    return stores[session_id]

def get_3k_store(session_id : str):
    K = 3
    if session_id not in stores:    # 아직 대화를 한번도 나눈적이 없는 경우
        stores[session_id] = InMemoryChatMessageHistory()

    hist = stores.setdefault(session_id, InMemoryChatMessageHistory())

    if len(hist.messages) > K:
        hist.messages[:] = hist.messages[-K:]

    return hist

def create_history_chain(chain):
    # with_history = RunnableWithMessageHistory(
    #     chain,
    #     lambda sid: get_store(sid),
    #     input_messages_key="question",
    #     history_messages_key="history"
    # )
    
    with_history = RunnableWithMessageHistory(
        chain,
        get_3k_store,
        input_messages_key="question",
        history_messages_key="history"
    )
    return with_history

prompt = create_prompt()
model = create_model()
chain = create_chain(prompt, model)
with_history = create_history_chain(chain)
cfg = {"configurable" : {"session_id" : "user-123"}}

def mychat(text):
    return with_history.invoke({
        "question" : text
    }, config=cfg)

if __name__ == "__main__":
    for token in mychat("안녕?"):
        if token is None:
            continue
        
        # 토큰을 웹소켓으로 전송
        print(token, end="", flush=True) # flush=True 추가
        