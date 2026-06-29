import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Windows 한국어 환경(cp949)에서 이모지 출력 에러 방지
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 환경변수 로드
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent import PaperResearcherAgent

st.set_page_config(page_title="AI 논문 리서치 팀", page_icon="📚", layout="wide")

st.title("📚 AI 논문 리서치 팀 (Supervisor Workflow)")
st.markdown("""
이 시스템은 **팀장(라우터), 연구원(검색), 요약가(요약), 번역가(번역), 검수자(평가)** 총 5명의 AI 에이전트가 협업하여 
논문을 완벽하게 요약/번역하는 Multi-Agent 상태 머신입니다.
""")

if "agent" not in st.session_state:
    st.session_state.agent = PaperResearcherAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 사이드바: 논문 URL 입력 ---
with st.sidebar:
    st.header("🔗 ArXiv 논문 주소 입력")
    st.markdown("분석하고 싶은 ArXiv 논문 주소를 입력하세요. (abs, pdf 링크 모두 가능)")
    sidebar_url = st.text_input("예: https://arxiv.org/abs/2606.28270")
    sidebar_submit = st.button("주소로 분석 시작")

# --- 메인 채팅창 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기 (사이드바 버튼 클릭이거나, 채팅창 입력이거나 둘 중 하나면 실행)
chat_prompt = st.chat_input("검색할 논문 키워드나 URL을 직접 입력하세요.")

if sidebar_submit and sidebar_url:
    prompt = sidebar_url
elif chat_prompt:
    prompt = chat_prompt
else:
    prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 팀장(Supervisor)이 팀원들을 지휘하는 중...", expanded=True) as status:
            
            final_state = {}
            # 🚀 LangGraph 스트리밍 실행: 노드별 실시간 진행 상황 표시
            for node_name, state_update, accumulated in st.session_state.agent.stream(prompt):
                final_state = accumulated
                
                if node_name == "supervisor":
                    next_node = state_update.get("next_node", "?")
                    st.write(f"👨‍💼 **[Supervisor]** 다음 작업자 지정 → **{next_node}**")
                
                elif node_name == "researcher":
                    raw = state_update.get("raw_paper_content", "")
                    title_line = raw.split('\\n')[0]
                    # Full Paper Content가 있으면 페이지 수와 글자 수 표시
                    if "[Full Paper Content]" in raw:
                        full_text = raw.split("[Full Paper Content]\\n")[-1]
                        page_count = full_text.count('\\n\\n') + 1  # 대략적 페이지 수
                        char_count = len(full_text)
                        st.write(f"🔍 **[Researcher]** {title_line}")
                        st.success(f"📄 논문 원문 전체 파싱 완료: **약 {char_count:,}자** 추출")
                    else:
                        st.write(f"🔍 **[Researcher]** {title_line}")
                        st.warning("⚠️ Abstract만 가져옴 (Full-Text 파싱 실패)")
                
                elif node_name == "summarizer":
                    summary = state_update.get("summarized_content", "")
                    rev = state_update.get("summary_revision_count", 1)
                    if rev == 1:
                        st.write(f"✍️ **[Summarizer]** 심층 영어 요약 완료 ({len(summary):,}자)")
                    else:
                        st.write(f"✍️ **[Summarizer]** {rev}차 요약(Fact Correction) 완료 ({len(summary):,}자)")
                        
                elif node_name == "summary_fact_checker":
                    feedback = state_update.get("summary_fact_feedback", "")
                    if "PASS" in feedback.upper():
                        st.write("🧐 **[Summary Fact Critic]** ✅ 요약 팩트체크 **통과(PASS)**!")
                    else:
                        st.write(f"🧐 **[Summary Fact Critic]** ❌ 환각 발견: _{feedback}_")
                        
                elif node_name == "terminology_extractor":
                    glossary = state_update.get("dynamic_glossary", {})
                    num_rules = len(glossary.get("rules", []))
                    st.write(f"🔤 **[Terminology Extractor]** 📚 {num_rules}개의 핵심 용어 번역 금지 규칙 추출 완료")
                
                elif node_name == "translator":
                    rev = state_update.get("revision_count", 1)
                    translation = state_update.get("korean_translation", "")
                    if rev == 1:
                        st.write(f"🌐 **[Translator]** 1차 한글 번역 완료 ({len(translation):,}자)")
                    else:
                        st.write(f"🌐 **[Translator]** {rev}차 재번역(Reflexion) 완료 ({len(translation):,}자)")
                
                elif node_name == "critic":
                    feedback = state_update.get("critic_feedback", "")
                    if "PASS" in feedback.upper():
                        st.write("🧐 **[Critic]** ✅ 번역 퀄리티 **통과(PASS)**!")
                    else:
                        st.write(f"🧐 **[Critic]** ❌ 반려: _{feedback}_")
            
            status.update(label="✅ 최종 번역 리포트 완성!", state="complete", expanded=False)
            
        final_translation = final_state.get("korean_translation", "결과를 가져오지 못했습니다.")
        english_summary = final_state.get("summarized_content", "")
        
        st.markdown("### 🇰🇷 최종 한글 번역본")
        st.info(final_translation)
        
        with st.expander("원본 영어 요약 보기"):
            st.write(english_summary)
            
        st.session_state.messages.append({"role": "assistant", "content": f"### 🇰🇷 최종 한글 번역본\\n{final_translation}"})
