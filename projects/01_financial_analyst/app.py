import streamlit as st
import time
import os
import sys
from dotenv import load_dotenv

# 상위 폴더의 .env 파일에 있는 API 키 로드
load_dotenv()

# 프로젝트 루트 경로를 path에 추가하여 core 모듈을 불러올 수 있게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.agent import FinancialAgent

st.set_page_config(page_title="AI Financial Analyst", page_icon="📈", layout="wide")

st.title("📈 AI Financial Analyst Agent (Real Data)")
st.markdown("""
실제 기업의 10-K 보고서(PDF)와 차트 이미지(PNG/JPG)를 업로드하고 질문해보세요.
LangGraph 기반의 자율 에이전트가 문서를 분석하고 결과를 검증하여 완벽한 리포트를 작성합니다.
""")

# 세션 상태 초기화
if "agent" not in st.session_state:
    st.session_state.agent = FinancialAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 사이드바: 파일 업로드 기능 ---
with st.sidebar:
    st.header("📂 데이터 업로드")
    
    # 1. PDF 업로드
    uploaded_pdf = st.file_uploader("재무제표 보고서 (PDF)", type=["pdf"])
    if uploaded_pdf:
        # data 폴더에 임시 저장
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        # 버튼을 누르면 RAG 엔진에 인덱싱
        if st.button("문서 분석 시작 (RAG 인덱싱)"):
            with st.spinner("PDF를 청크로 쪼개고 벡터 DB에 저장 중입니다..."):
                st.session_state.agent.load_pdf(pdf_path)
            st.success("문서 인덱싱 완료!")
            
    st.divider()
    
    # 2. 이미지 업로드
    uploaded_image = st.file_uploader("주가 차트 이미지 (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        os.makedirs("data", exist_ok=True)
        image_path = os.path.join("data", uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        st.session_state.agent.load_image(image_path)
        st.image(image_path, caption="업로드된 차트", use_container_width=True)


# --- 메인 채팅창 ---
# 대화 기록 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기
if prompt := st.chat_input("재무제표 요약이나 주가 차트 분석을 요청해보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 에이전트가 생각하는 중...", expanded=True) as status:
            st.write("1. 🚦 라우터(Router) 판단 중...")
            
            result = st.session_state.agent.analyze(prompt)
            route = result.get("route")
            
            if route == "pdf":
                st.write("👉 [분기] **RAG 노드** 선택: 실제 벡터 DB에서 문서 검색을 진행했습니다.")
            else:
                st.write("👉 [분기] **Vision 노드** 선택: 업로드된 이미지를 GPT-4o 멀티모달 API로 전송했습니다.")
                
            st.write(f"2. 🔍 컨텍스트(Context) 추출 완료.")
            
            rev_count = result.get("revision_count", 1)
            if rev_count > 1:
                st.write(f"3. 🔁 비평가(Critic)의 지적을 받아 총 **{rev_count}번** 재작성(Reflection Loop) 하였습니다.")
            else:
                st.write("3. ✨ 첫 번째 초안이 비평가(Critic)의 검증을 통과했습니다.")
                
            status.update(label="✅ 분석 완료!", state="complete", expanded=False)
            
        final_analysis = result.get("analysis", "분석 결과를 가져오지 못했습니다.")
        st.markdown("### 📊 최종 분석 리포트")
        st.info(final_analysis)
        
        st.session_state.messages.append({"role": "assistant", "content": f"### 📊 최종 분석 리포트\\n{final_analysis}"})
