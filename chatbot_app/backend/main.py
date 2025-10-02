# uv add "fastapi[all]"
# 서버 실행: uvicorn main:app --port 8000 --reload
from fastapi import WebSocket 
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.models.chatgpt import with_history, cfg

from fastapi import FastAPI, WebSocket 

app = FastAPI()

@app.get("/")
def home():
    return {"hello": "world"}

# 웹소켓
from pydantic import BaseModel
class Request(BaseModel):
    question: str

@app.websocket("/ws/streaming")
async def websocket_text(websocket: WebSocket):
    await websocket.accept()

    try:
        # 데이터 받기 
        data = await websocket.receive_json()
        req = Request(**data)

        print("question: ")
        print(req.question)
        
        # AI 활동
        print("answer: ")
        async for token in with_history.astream({
            "question" : req.question
        }, config=cfg):
            if token is None:
                continue
            
            # 토큰을 웹소켓으로 전송
            print(token, end="", flush=True)
            await websocket.send_text(token)
            
        print()
        await websocket.send_text("[END]")
    
    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    finally:
        await websocket.close()
        print("WebSocket 연결 종료")