"""
LangGraph의 상태(State)와 노드(Nodes), 엣지(Edges)를 정의하고 컴파일하는 모듈입니다.
"""

from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from .rag_engine import RAGEngine
from .vision_engine import VisionEngine

# 1. State 정의
class AgentState(TypedDict):
    question: str
    route: str           
    context: str         
    analysis: str        
    feedback: str        
    revision_count: int  

# 2. Graph Builder 클래스
class FinancialGraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.rag = RAGEngine()
        self.vision = VisionEngine(llm)
        self.workflow = StateGraph(AgentState)
        self._build_graph()
        
    def _build_graph(self):
        # 노드 등록
        self.workflow.add_node("router", self.router_node)
        self.workflow.add_node("rag", self.rag_node)
        self.workflow.add_node("vision", self.vision_node)
        self.workflow.add_node("analyzer", self.analyzer_node)
        self.workflow.add_node("reflection", self.reflection_node)
        
        # 엣지 연결
        self.workflow.set_entry_point("router")
        self.workflow.add_conditional_edges(
            "router",
            self.route_after_router,
            {"rag": "rag", "vision": "vision"}
        )
        self.workflow.add_edge("rag", "analyzer")
        self.workflow.add_edge("vision", "analyzer")
        self.workflow.add_edge("analyzer", "reflection")
        
        # 순환 루프 엣지
        self.workflow.add_conditional_edges(
            "reflection",
            self.route_after_reflection,
            {"retry": "analyzer", "end": END}
        )
        
        # 컴파일
        self.app = self.workflow.compile()
        
    def compile(self):
        return self.app

    # --- 노드 함수들 ---
    def router_node(self, state: AgentState):
        prompt = ChatPromptTemplate.from_template(
            "다음 질문이 '차트/그래프/이미지'를 분석해야 하는 질문이면 'image', '재무제표/보고서/텍스트'를 검색해야 하는 질문이면 'pdf'라고만 대답해.\\n"
            "질문: {question}"
        )
        result = (prompt | self.llm).invoke({"question": state["question"]}).content.strip().lower()
        route = "image" if "image" in result else "pdf"
        return {"route": route}
        
    def rag_node(self, state: AgentState):
        context = self.rag.search(state["question"])
        return {"context": context}
        
    def vision_node(self, state: AgentState):
        image_path = getattr(self, "current_image_path", None)
        context = self.vision.analyze_image(image_path, state["question"])
        return {"context": context}
        
    def analyzer_node(self, state: AgentState):
        feedback_msg = f"\\n\\n이전 피드백: {state['feedback']}" if state.get('feedback') else ""
        prompt = ChatPromptTemplate.from_template(
            "당신은 월스트리트 수석 펀드매니저입니다. 다음 맥락을 바탕으로 사용자의 질문에 대한 분석 리포트를 3문장으로 작성하세요."
            "{feedback_msg}\\n\\n맥락: {context}\\n질문: {question}"
        )
        analysis = (prompt | self.llm).invoke({
            "context": state["context"], 
            "question": state["question"], 
            "feedback_msg": feedback_msg
        }).content
        new_count = state.get('revision_count', 0) + 1
        return {"analysis": analysis, "revision_count": new_count}
        
    def reflection_node(self, state: AgentState):
        prompt = ChatPromptTemplate.from_template(
            "당신은 깐깐한 편집장입니다. 다음 분석 리포트에 '구체적인 숫자나 수치'가 포함되어 있는지 확인하세요.\\n"
            "숫자가 포함되어 완벽하다면 'PASS'라고만 응답하고, 부족하다면 어떤 점을 보완해야 할지 1문장으로 피드백하세요.\\n\\n"
            "리포트: {analysis}"
        )
        feedback = (prompt | self.llm).invoke({"analysis": state["analysis"]}).content
        return {"feedback": feedback}
        
    # --- 엣지 라우팅 함수들 ---
    def route_after_router(self, state: AgentState):
        return "rag" if state["route"] == "pdf" else "vision"
        
    def route_after_reflection(self, state: AgentState):
        if "PASS" in state["feedback"] or state["revision_count"] >= 3:
            return "end"
        else:
            return "retry"
