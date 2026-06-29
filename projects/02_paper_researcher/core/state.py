from typing import TypedDict, Optional

class TeamState(TypedDict):
    question: str                  # 사용자의 최초 질문 또는 지시사항
    next_node: str                 # 다음으로 라우팅될 에이전트 이름
    raw_paper_content: str         # 검색된 영어 원문
    summarized_content: str        # 영어 요약본
    korean_translation: str        # 한글 번역본
    critic_feedback: str           # 검수자의 피드백 (반려 시 재작성 요구사항)
    revision_count: int            # 반려(빠꾸)로 인해 번역을 다시 한 횟수
    
    # [NEW] 요약 환각 방지(Fact Checker) 관련 신규 필드
    structured_summary: dict       # 구조화된 요약 데이터
    summary_fact_feedback: str     # 요약 검수자의 피드백 (팩트체크)
    summary_revision_count: int    # 요약 재작성 횟수
    
    # [NEW] 동적 용어집(Dynamic Glossary) 필드
    dynamic_glossary: dict         # 논문 맞춤형 용어 번역 규칙
