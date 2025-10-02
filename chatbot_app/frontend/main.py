import streamlit as st
import sys, os
import asyncio
import websockets
import json

WEBSOCKET_URL = "ws://localhost:8000/ws/streaming"

AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # 없으면 None
EXTRA_HEADERS = {}
if AUTH_TOKEN:
    EXTRA_HEADERS["Authorization"] = f"Bearer {AUTH_TOKEN}"
EXTRA_HEADERS["X-Client"] = "frontend"

async def send_message(question):
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            # FastAPI에게 메시지 전송
            json_data = json.dumps(
                {"question" : question},
                ensure_ascii=False
            )
            await websocket.send(json_data)

            # FastAPI 서버에서 응답 받기
            while True:
                token = await websocket.recv()
                if token == "[END]":
                    break
                yield token
    except Exception as e:
        print(f"[WS] Unexpected error: {repr(e)}")
        raise

# ================================================
# streamlit 파트
# ================================================

if not "messages" in st.session_state:
    st.session_state["messages"] = []

history = st.session_state["messages"]

# 프로필 설정
profile = {
    "user": "app/resources/sample.png",
    "ai"  : "app/resources/chatbot.png"
}

# 챗봇 제목 
st.title("챗봇 만들기")

# 과거 메시지 출력 
if len(st.session_state["messages"]) > 0:
    for chat in st.session_state["messages"]:
        name = chat["role"]
        avatar = profile[name]
        st.chat_message(name=name, avatar=avatar).markdown(chat["content"])

# 사용자 입력
input_text = st.chat_input("메세지를 입력하세요...")

# 사용자 입력 이후
if input_text:
    st.chat_message(name="user", avatar = profile['user']).markdown(input_text)
    st.session_state["messages"].append({"role": "user", "content": input_text})

    with st.chat_message(name="ai", avatar = profile["ai"]):
        container = st.empty() # 빈 자리 맡기
        with container:
            with st.spinner("생각하는 중이에요..."):
                full_response  = ""
                # answer = mychat(input_text)
                def stream_response_sync():
                    # 새 이벤트 루프를 생성하고 send_message 코루틴 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async_gen = send_message(input_text)
                    
                    while True:
                        try:
                            token = loop.run_until_complete(async_gen.__anext__())
                            yield token
                        except StopAsyncIteration:
                            break
                        except websockets.exceptions.ConnectionClosedOK:
                            break
                        except Exception as e:
                            print(f"Streaming Error: {e}")
                            break
                    loop.close()

                # Streamlit의 write_stream을 사용하여 스트리밍 출력
                for chunk in st.write_stream(stream_response_sync):
                    full_response += chunk

                answer = full_response
            st.markdown(answer)

    # 대화 저장
    st.session_state["messages"].append({"role": "ai", "content": answer})