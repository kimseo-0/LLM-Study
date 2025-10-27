### MCP 란?

- 기본적으로 AI 와 외부 시스템간의 다리역할을 하는 연결 표준 방식

1. 아키텍처 구성요소
    - MCP Server : 데이터/도구/프롬프트 를 제공하는 서버
    - MCP Client : 서버와 통신하는 클라이언트(중계 역할 : gateway)
    - MCP Host   : 실제 애플리케이션(직접 만든 langgraph agent)

2. 주요 기능
    - Resources : 파일, 데이터베이스
    - Tools : 도구
    - Prompts : 해보니까 성능 잘뽑아내는 프롬프트가 있을 경우 제공
    (예 : few shot 제공)