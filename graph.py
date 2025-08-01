import os
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    """그래프 상태를 정의하는 타입"""
    messages: Annotated[List[BaseMessage], add_messages]


# 기본 설정
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can use tools to answer questions."

def build_graph(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: List = None
):
    """
    LangGraph 애플리케이션 빌드
    
    Args:
        model: 사용할 모델명
        temperature: 모델의 temperature 설정
        system_prompt: 시스템 프롬프트
    
    Returns:
        컴파일된 그래프
    """
    
    # LLM 생성
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    
    # 챗봇 노드 정의
    def chatbot_node(state: State) -> State:
        response = llm.invoke(
            [SystemMessage(content=system_prompt)] +
            state["messages"]
        )
        return {"messages": [response]}
    
    
    # 그래프 빌더 생성
    builder = StateGraph(State)
    
    # 노드 추가
    builder.add_node("chatbot", chatbot_node)
    
    # 엣지 추가
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)

graph = build_graph()