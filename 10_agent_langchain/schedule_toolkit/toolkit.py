from typing import List, Dict
from langchain_core.tools import BaseTool, BaseToolkit
# from langchain.agents.agent_toolkits import BaseToolKit

from .tools import AddToDoTool, GetToDoTool, DeleteToDoTool

class ScheduleToolkit(BaseToolkit):
    """스케쥴 관리를 위한 툴킷"""
    def get_tools(self) -> List[BaseTool]:
        return [
            AddToDoTool(),
            GetToDoTool(),
            DeleteToDoTool()
        ]