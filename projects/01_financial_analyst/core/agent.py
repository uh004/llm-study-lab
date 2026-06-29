"""
외부(UI 등)에서 쉽게 호출할 수 있도록 LangGraph를 감싸는 파사드(Facade) 모듈입니다.
"""

from langchain_openai import ChatOpenAI
from .graph import FinancialGraphBuilder

class FinancialAgent:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.builder = FinancialGraphBuilder(self.llm)
        self.app = self.builder.compile()
        self.current_image_path = None
        
    def load_pdf(self, pdf_path: str):
        """UI에서 업로드된 PDF를 RAG 엔진에 로드합니다."""
        self.builder.rag.load_and_index_pdf(pdf_path)
        
    def load_image(self, image_path: str):
        """UI에서 업로드된 이미지를 저장해둡니다."""
        self.current_image_path = image_path
        
    def analyze(self, question: str) -> dict:
        """
        사용자의 질문을 받아 LangGraph 워크플로우를 실행하고 결과를 반환합니다.
        """
        print(f"\\n🚀 [FinancialAgent] 분석 시작: '{question}'")
        
        # 현재 저장된 image_path를 vision_node가 사용할 수 있도록 builder 객체에 주입하거나 state에 넣을 수도 있습니다.
        # graph.py의 vision_node가 self.vision.analyze_image("dummy_path", ...) 로 되어있으므로,
        # 에이전트 단위에서 임시로 보관한 경로를 넘겨주기 위해 약간의 해킹을 하거나 state를 수정해야 하지만,
        # 가장 간단한 방법은 builder의 멤버 변수로 전달하는 것입니다.
        self.builder.current_image_path = self.current_image_path
        
        inputs = {"question": question}
        result = self.app.invoke(inputs)
        return {
            "route": result.get("route"),
            "context": result.get("context"),
            "analysis": result.get("analysis"),
            "feedback": result.get("feedback"),
            "revision_count": result.get("revision_count")
        }
