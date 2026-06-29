from langgraph.graph import StateGraph, END
from .state import TeamState
from .nodes import ResearchTeamNodes

class ResearchGraphBuilder:
    def __init__(self, llm):
        self.nodes = ResearchTeamNodes(llm)
        self.workflow = StateGraph(TeamState)
        self._build_graph()

    def route_after_supervisor(self, state: TeamState):
        return state["next_node"]

    def route_after_summary_fact_checker(self, state: TeamState):
        if state["summary_fact_feedback"] == "PASS":
            return "pass"
        else:
            return "retry_summary"

    def route_after_critic(self, state: TeamState):
        if state["critic_feedback"] == "PASS":
            return "end"
        else:
            return "retry_translation"

    def _build_graph(self):
        # 1. 노드 추가
        self.workflow.add_node("supervisor", self.nodes.supervisor_node)
        self.workflow.add_node("researcher", self.nodes.researcher_node)
        self.workflow.add_node("summarizer", self.nodes.summarizer_node)
        self.workflow.add_node("summary_fact_checker", self.nodes.summary_fact_checker_node)
        self.workflow.add_node("terminology_extractor", self.nodes.terminology_extractor_node)
        self.workflow.add_node("translator", self.nodes.translator_node)
        self.workflow.add_node("critic", self.nodes.critic_node)

        # 2. 진입점 설정
        self.workflow.set_entry_point("supervisor")

        # 3. 엣지(Edge) 연결
        self.workflow.add_conditional_edges(
            "supervisor",
            self.route_after_supervisor,
            {"researcher": "researcher", "summarizer": "summarizer"}
        )
        self.workflow.add_edge("researcher", "summarizer")
        self.workflow.add_edge("summarizer", "summary_fact_checker")
        
        # 3.5. 요약 팩트체크(환각 방지) 루프
        self.workflow.add_conditional_edges(
            "summary_fact_checker",
            self.route_after_summary_fact_checker,
            {
                "retry_summary": "summarizer",           # 환각 발견 시 요약가에게 리턴
                "pass": "terminology_extractor"          # 통과 시 용어 추출기로 진행
            }
        )
        
        self.workflow.add_edge("terminology_extractor", "translator")
        self.workflow.add_edge("translator", "critic")

        # 4. 검증 루프(Reflexion) 조건부 엣지
        self.workflow.add_conditional_edges(
            "critic",
            self.route_after_critic,
            {
                "retry_translation": "translator",  # 반려 시 번역가에게 리턴
                "end": END                          # 통과 시 워크플로우 종료
            }
        )

        self.app = self.workflow.compile()

    def compile(self):
        return self.app
