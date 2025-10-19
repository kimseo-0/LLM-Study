from typing import List, Type
from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, Field

# 스케쥴 등록하는 리스트
schedule: List[str] = []
complete_schedule: List[str] = []
# 1. 스케쥴 등록 도구
# 2. 스케쥴 확인 도구

# 1.1 스케줄 스키마 설정
class ToDoInput(BaseModel): # pydantic
    item: str = Field(description="오늘의 스케쥴 항목")

# 1.2 스케줄 등록 도구 설정
class AddToDoTool(BaseTool):
    name : str = "add_todo"
    description : str = "오늘 스케줄에 새 항목을 추가합니다"
    args_schema : Type[BaseModel] = ToDoInput

    def _run(self, item: str):
        schedule.append(item)
        return f"{item}이 스케줄에 등록 되었습니다"

class DeleteToDoTool(BaseTool):
    name : str = "cancel_todo"
    description : str = "오늘 스케줄을 취소합니다"
    args_schema : Type[BaseModel] = ToDoInput

    def _run(self, item: str):
        if item in schedule:
            schedule.remove(item)
            return f"{item}이 스케줄에 삭제 되었습니다"
        else:
            return f"{item}은 스케줄에 없습니다"

class DeleteToDoTool(BaseTool):
    name : str = "complete_todo"
    description : str = "오늘 스케줄을 완료합니다"
    args_schema : Type[BaseModel] = ToDoInput

    def _run(self, item: str):
        if item in schedule:
            schedule.remove(item)
            complete_schedule.append(item)
            return f"{item}을 완료했습니다"
        else:
            return f"{item}은 스케줄에 없습니다"


class GetToDoTool(BaseTool):
    name : str = "get_todo"
    description : str = "오늘 스케줄을 확인합니다"

    def _run(self):
        if not schedule:
            return "할일이 없어요"
        all_schedule = "\n".join(schedule)
        return f"할일 목록은 : {all_schedule}"