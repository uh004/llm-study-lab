from langchain_openai import ChatOpenAI
from .graph import ResearchGraphBuilder

class PaperResearcherAgent:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.app = ResearchGraphBuilder(self.llm).compile()
        
    def run(self, question: str) -> dict:
        """
        웹 프론트엔드(Streamlit 등)에서 쉽게 호출할 수 있는 메인 실행 함수
        """
        print(f"\\n🚀 [PaperResearcherAgent] 팀장 워크플로우 시작: '{question}'")
        inputs = {"question": question}
        result = self.app.invoke(inputs)
        return result
    
    def stream(self, question: str):
        """
        LangGraph의 stream()을 사용하여 노드별 실행 결과를 실시간으로 yield합니다.
        각 yield 값은 (node_name, state_update) 형태입니다.
        """
        print(f"\\n🚀 [PaperResearcherAgent] 팀장 워크플로우 시작 (스트리밍): '{question}'")
        inputs = {"question": question}
        final_state = {}
        for event in self.app.stream(inputs):
            for node_name, state_update in event.items():
                final_state.update(state_update)
                yield node_name, state_update, final_state
