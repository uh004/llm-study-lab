import arxiv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from typing import Optional
from .state import TeamState

class EvidenceItem(BaseModel):
    claim: str = Field(description="핵심 주장 또는 수치를 2~3문장으로 상세하게 기술. 구체적인 기술적 세부사항, 수치, 방법론을 반드시 포함할 것.")
    evidence_quote: str = Field(description="주장을 뒷받침하는 원문의 정확한 인용구 (짧게)")
    uncertainty_notes: Optional[str] = Field(description="명시되지 않고 추론된 내용일 경우 여기에 기재")
    page_number: Optional[str] = Field(description="해당 인용구가 위치한 페이지 번호 (예: [Page 3])")

class StructuredPaperSummary(BaseModel):
    background_motivation: list[EvidenceItem] = Field(description="연구 배경 및 동기 (3~5개의 핵심 주장)")
    core_methodology: list[EvidenceItem] = Field(description="핵심 방법론 및 아키텍처 설계 (3~5개의 핵심 주장)")
    experimental_setup: list[EvidenceItem] = Field(description="실험 환경, 데이터셋, 하이퍼파라미터 (3~5개의 핵심 주장)")
    key_results: list[EvidenceItem] = Field(description="주요 실험 결과 및 성능 수치 (3~5개의 핵심 주장)")
    limitations_future_work: list[EvidenceItem] = Field(description="한계점 및 향후 연구 방향 (논문에 명시된 것만)")
    significance_impact: list[EvidenceItem] = Field(description="학술적 의의 및 후속 연구에 미친 영향 (3~5개의 핵심 주장)")

class SummaryFactEvaluation(BaseModel):
    is_hallucinated: bool = Field(description="원문에 없는 주장, 수치 오류, 지어낸 한계점이 존재하는가? (True/False)")
    hallucination_feedback: Optional[str] = Field(description="is_hallucinated가 True일 경우 1문장 구체적 지적사항 (어떤 수치가 틀렸는지 등)")

class TerminologyRule(BaseModel):
    english_term: str = Field(description="영어 원문의 핵심 전문 용어 (예: Attention, Zero-shot 등)")
    forbidden_translations: list[str] = Field(description="번역가가 쓰면 안 되는 직역/오역 리스트 (예: ['주의', '주의 메커니즘'])")

class DynamicGlossary(BaseModel):
    rules: list[TerminologyRule] = Field(description="논문의 핵심 전문 용어 번역 금지 규칙 목록")

class CriticEvaluation(BaseModel):
    is_missing_sentences: bool = Field(description="영어 원문의 핵심 문장이 번역에서 통째로 누락되었는가? (단어 번역 여부는 절대 해당 안됨. 누락되었다면 True, 아니면 False)")
    is_grammar_broken: bool = Field(description="한국어 문법이 완전히 붕괴되어 전혀 읽을 수 없는 문장이 있는가? (영어가 섞여있는 것은 절대 해당 안됨. 붕괴되었다면 True, 아니면 False)")

class ResearchTeamNodes:
    def __init__(self, llm):
        self.llm = llm

    def supervisor_node(self, state: TeamState):
        """팀장 에이전트: 상태를 보고 다음 작업자를 지정합니다."""
        print(f"\\n👨‍💼 [Supervisor] 업무 분배 중... (요청: '{state['question']}')")
        if not state.get("raw_paper_content"):
            return {"next_node": "researcher"}
        else:
            return {"next_node": "summarizer"}

    def researcher_node(self, state: TeamState):
        """연구원 에이전트: 논문을 검색해오거나 업로드된 PDF를 파싱합니다."""
        print("🔍 [Researcher] 논문 검색 및 파싱 중...")
        question = state["question"]
        
        # 1. PDF 업로드 방식인지 확인 (간단한 식별: 질문이 '.pdf'로 끝나는 파일 경로 형태인 경우)
        if question.lower().endswith('.pdf'):
            try:
                loader = PyPDFLoader(question)
                docs = loader.load()
                # 🚀 제한 없이 논문 전체 페이지 텍스트 추출 (Full-Text)
                full_text = "\\n\\n".join([f"[Page {i+1}]\\n{doc.page_content}" for i, doc in enumerate(docs)])
                print(f"   📄 PDF 파싱 완료: 총 {len(docs)}페이지, {len(full_text):,}자 추출")
                return {"raw_paper_content": f"Title: Uploaded PDF ({question})\\n\\n[Full Paper Content]\\n{full_text}"}
            except Exception as e:
                return {"raw_paper_content": f"PDF 파싱 실패: {e}"}
        
        # 2. ArXiv URL 방식인지 확인 (예: https://arxiv.org/abs/2606.28270 또는 /pdf/...)
        import re
        arxiv_url_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+(v\d+)?)', question)
        if arxiv_url_match:
            arxiv_id = arxiv_url_match.group(1)
            try:
                # ArXiv API로 메타데이터(제목, 저자) 가져오기
                client = arxiv.Client()
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(client.results(search))
                print(f"   📋 논문 메타데이터 확보: {paper.title}")
                
                # 🚀 PDF URL을 직접 PyPDFLoader에 넘겨서 전체 원문 파싱
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                print(f"   📥 PDF 원문 다운로드 중: {pdf_url}")
                loader = PyPDFLoader(pdf_url)
                docs = loader.load()
                full_text = "\\n\\n".join([f"[Page {i+1}]\\n{doc.page_content}" for i, doc in enumerate(docs)])
                print(f"   📄 PDF 파싱 완료: 총 {len(docs)}페이지, {len(full_text):,}자 추출")
                
                content = f"Title: {paper.title}\\nAuthors: {', '.join([a.name for a in paper.authors])}\\n\\n[Full Paper Content]\\n{full_text}"
                return {"raw_paper_content": content}
            except Exception as e:
                print(f"   ⚠️ Full-Text 파싱 실패, Abstract로 대체: {e}")
                # Full-text 실패 시 Abstract만이라도 가져오기 (Fallback)
                try:
                    client = arxiv.Client()
                    search = arxiv.Search(id_list=[arxiv_id])
                    paper = next(client.results(search))
                    content = f"Title: {paper.title}\\nAuthors: {', '.join([a.name for a in paper.authors])}\\nAbstract: {paper.summary}"
                    return {"raw_paper_content": content}
                except:
                    return {"raw_paper_content": f"ArXiv URL 파싱 실패: {e}"}
        
        # 3. 일반 키워드 검색 방식 (ArXiv)
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=question,
                max_results=1,
                sort_by=arxiv.SortCriterion.Relevance
            )
            paper = next(client.results(search))
            content = f"Title: {paper.title}\\nAuthors: {', '.join([a.name for a in paper.authors])}\\nAbstract: {paper.summary}"
            return {"raw_paper_content": content}
        except Exception as e:
            # 검색 실패 시 Fallback (모의 데이터)
            mock_paper = "Title: Transformer (Fallback)\\nAbstract: Attention is all you need."
            return {"raw_paper_content": mock_paper}

    def _format_structured_summary(self, summary: StructuredPaperSummary) -> str:
        sections = {
            "Background & Motivation": summary.background_motivation,
            "Core Methodology & Architecture": summary.core_methodology,
            "Experimental Setup & Datasets": summary.experimental_setup,
            "Key Results & Performance": summary.key_results,
            "Limitations & Future Work": summary.limitations_future_work,
            "Significance & Impact": summary.significance_impact
        }
        
        formatted = ""
        for i, (title, items) in enumerate(sections.items(), 1):
            formatted += f"### {i}. {title}\\n"
            for item in items:
                formatted += f"- {item.claim}\\n"
                formatted += f"  > **Evidence:** \\\"{item.evidence_quote}\\\" {item.page_number if item.page_number else ''}\\n"
                if item.uncertainty_notes:
                    formatted += f"  > **Note:** {item.uncertainty_notes}\\n"
            formatted += "\\n"
        return formatted

    def summarizer_node(self, state: TeamState):
        """요약가 에이전트: 방대한 논문을 구조화하여 요약합니다."""
        print("✍️ [Summarizer] 원문 구조화 요약 중...")
        
        feedback = state.get("summary_fact_feedback", "")
        feedback_msg = f"\\n\\n[🚨 Fact Checker Feedback 🚨]\\nYour previous summary failed fact-checking. Fix these issues: {feedback}\\n" if feedback and feedback != "PASS" else ""
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert AI academic researcher writing a detailed review for a top-tier conference. "
            "Provide a highly detailed and professional summary of the following paper content using the structured output format.\\n\\n"
            "[DEPTH REQUIREMENTS]\\n"
            "- For EACH of the 6 sections, provide 3 to 5 detailed claims (EvidenceItem).\\n"
            "- Each claim MUST be 2-3 sentences long with specific technical details, numbers, and methodology descriptions.\\n"
            "- Do NOT write single-sentence summaries. Be thorough and comprehensive.\\n\\n"
            "[CRITICAL ANTI-HALLUCINATION RULES]\\n"
            "1. Do NOT invent limitations.\\n"
            "2. Only include limitations or future work explicitly stated by the authors. If the paper does not explicitly discuss a limitation, write that the paper does not explicitly discuss it.\\n"
            "3. Every numerical claim must be grounded in the paper.\\n"
            "4. If a claim is inferred rather than explicitly stated, put it in uncertainty_notes.\\n"
            "5. For each major claim, provide a short evidence_quote from the paper.\\n"
            "6. Use the [Page X] markers in the text to fill in the page_number field.\\n\\n"
            "Paper: {paper}{feedback_msg}"
        )
        
        structured_llm = self.llm.with_structured_output(StructuredPaperSummary)
        structured_summary = (prompt | structured_llm).invoke({
            "paper": state["raw_paper_content"],
            "feedback_msg": feedback_msg
        })
        
        formatted_summary = self._format_structured_summary(structured_summary)
        
        count = state.get("summary_revision_count", 0) + 1
        return {
            "structured_summary": structured_summary.model_dump(),
            "summarized_content": formatted_summary,
            "summary_revision_count": count
        }

    def summary_fact_checker_node(self, state: TeamState):
        """팩트 체커 에이전트: 요약이 논문 원문에 기반하고 있는지(환각 방지) 검사합니다."""
        print("🧐 [Summary Fact Critic] 요약 내용 팩트체크 중...")
        
        prompt = ChatPromptTemplate.from_template(
            "You are a strict Fact Checker. Compare the structured summary with the original paper.\\n\\n"
            "[Evaluation Criteria]\\n"
            "1. Are there any claims in the summary that are NOT supported by the paper?\\n"
            "2. Are there any numerical errors (wrong parameters, dataset sizes, scores)?\\n"
            "3. Did the summarizer invent limitations or future works not explicitly stated in the paper?\\n"
            "4. Are the evidence_quotes actually found in the original text?\\n\\n"
            "Original Paper: {paper}\\n\\n"
            "Structured Summary: {summary}\\n"
        )
        
        structured_llm = self.llm.with_structured_output(SummaryFactEvaluation)
        eval_result = (prompt | structured_llm).invoke({
            "paper": state["raw_paper_content"],
            "summary": str(state["structured_summary"])
        })
        
        if not eval_result.is_hallucinated:
            feedback = "PASS"
        else:
            feedback = eval_result.hallucination_feedback if eval_result.hallucination_feedback else "Hallucination detected."
            print(f"   -> ❌ [팩트 오류 발견] {feedback}")
            
        return {"summary_fact_feedback": feedback}

    def terminology_extractor_node(self, state: TeamState):
        """용어 추출기 에이전트: 요약본을 바탕으로 논문의 전문 용어와 번역 금지어를 동적으로 추출합니다."""
        print("🔤 [Terminology Extractor] 동적 전문 용어집 구축 중...")
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert academic terminology extractor. Read the following paper summary and extract the most important technical terms.\\n"
            "For each technical term that SHOULD REMAIN IN ENGLISH (e.g., Attention, Transformer, ResNet, BLEU, Zero-shot), "
            "provide a list of 'forbidden_translations' (e.g., literal translations in Korean that sound awkward like '주의', '제로샷').\\n"
            "This will be used as a strict rulebook to prevent the translator from making literal translation mistakes.\\n\\n"
            "Paper Summary:\\n{summary}"
        )
        
        structured_llm = self.llm.with_structured_output(DynamicGlossary)
        glossary = (prompt | structured_llm).invoke({"summary": state["summarized_content"]})
        
        print(f"   -> 📚 {len(glossary.rules)}개의 핵심 용어 번역 규칙 추출 완료")
        
        return {"dynamic_glossary": glossary.model_dump()}

    def translator_node(self, state: TeamState):
        """번역가 에이전트: 영어를 학술적인 한글로 번역합니다."""
        print("🌐 [Translator] 한글 번역 중...")
        feedback = state.get("critic_feedback", "")
        prev_translation = state.get("korean_translation", "")
        
        if feedback and feedback != "PASS" and prev_translation:
            feedback_msg = (
                f"\\n\\n🚨 [긴급: 재번역 수정 지시사항] 🚨\\n"
                f"방금 당신이 번역한 결과물에서 팀장(Critic)이 다음과 같은 심각한 문제를 발견하여 반려했습니다.\\n"
                f"👉 지적사항: {feedback}\\n\\n"
                f"위 지적사항을 완벽하게 반영하여, 지적받은 용어나 문장을 올바르게 수정한 '새로운 전체 번역본'을 다시 작성하세요. (절대 이전과 똑같이 출력하지 마세요!)"
            )
        else:
            feedback_msg = ""
            
        glossary_dict = state.get("dynamic_glossary", {})
        glossary_rules = "없음"
        if glossary_dict and "rules" in glossary_dict:
            glossary_rules = "\\n".join([f"- '{rule['english_term']}': 절대 {rule['forbidden_translations']} 등으로 번역하지 마세요." for rule in glossary_dict["rules"]])
        
        prompt = ChatPromptTemplate.from_template(
            "당신은 네이처(Nature)나 사이언스(Science) 저널의 수석 학술 번역가입니다. "
            "다음 영어 요약본을 매우 전문적이고 격식 있는 한국어 학술 논문 톤(합니다/습니다 체)으로 번역하세요. "
            "단순한 요약이 아닌, 실제 연구원들이 읽을 수 있도록 원문의 깊이와 마크다운(Markdown) 서식을 그대로 살려서 작성하세요.\\n\\n"
            "[논문 전용 번역 금지 규칙 (Dynamic Glossary)]\\n"
            "다음 용어들은 절대로 한국어로 번역하지 말고 영어 원문 그대로 유지하세요!\\n"
            "{glossary_rules}\\n\\n"
            "==== [영어 원문] ====\\n{summary}\\n===================={feedback_msg}"
        )
        translation = (prompt | self.llm).invoke({
            "summary": state["summarized_content"], 
            "feedback_msg": feedback_msg,
            "glossary_rules": glossary_rules
        }).content
        
        count = state.get("revision_count", 0) + 1
        return {"korean_translation": translation, "revision_count": count}

    def critic_node(self, state: TeamState):
        """검수자 에이전트: 번역 품질을 평가하여 반려(빠꾸)할지 결정합니다."""
        print("🧐 [Critic] 번역 퀄리티 검수 중...")
        
        # 무한 루프 방지 (최대 3회까지만 재번역)
        if state.get("revision_count", 1) >= 3:
            print("   -> ⚠️ [강제 통과] 최대 재작성 횟수 초과.")
            return {"critic_feedback": "PASS"}
            
        translation = state.get("korean_translation", "")
        glossary_dict = state.get("dynamic_glossary", {})
        
        # 1. 결정론적(Deterministic) 용어 환각 1차 방어막 (모든 오류 한 번에 수집)
        violations = []
        if glossary_dict and "rules" in glossary_dict:
            for rule in glossary_dict["rules"]:
                for forbidden in rule["forbidden_translations"]:
                    if forbidden in translation:
                        violations.append(f"'{forbidden}' ➡️ '{rule['english_term']}'")
                        
        if violations:
            violation_msg = ", ".join(violations)
            print(f"   -> ❌ [결정론적 차단] 금지어 다수 발견됨: {violation_msg}")
            return {"critic_feedback": f"다음 금지된 표현들이 한 번에 발견되었습니다. 다음 규칙에 따라 모두 원문 그대로 수정하세요: [{violation_msg}]"}

        # 2. LLM 기반 문법 및 누락 검수
        structured_llm = self.llm.with_structured_output(CriticEvaluation)
        
        prompt = ChatPromptTemplate.from_template(
            "당신은 깐깐하지만 공정한 수석 연구원(팀장)입니다. 아래의 영어 원문과 한글 번역본을 비교하여 다음 2가지만 평가하세요.\\n\\n"
            "1. 원문의 핵심 문장이 통째로 누락되었는가? (is_missing_sentences)\\n"
            "2. 한국어 문법이 완전히 붕괴되어 전혀 읽을 수 없는 문장이 있는가? (is_grammar_broken)\\n\\n"
            "주의: 특정 단어를 번역하지 않고 영어로 남겨둔 것은 매우 잘한 일이므로 절대 문법 오류로 간주하지 마세요. "
            "오직 '문장이 통째로 날아간 경우'와 '말이 아예 안 되는 경우'에만 True를 반환하세요.\\n\\n"
            "[영어 원문]: {summary}\\n[한글 번역]: {translation}"
        )
        
        eval_result = (prompt | structured_llm).invoke({
            "summary": state["summarized_content"],
            "translation": state["korean_translation"]
        })
        
        # LLM에게 텍스트 작성 권한을 완전히 박탈하고 코드에서 제어
        if not eval_result.is_missing_sentences and not eval_result.is_grammar_broken:
            feedback = "PASS"
        else:
            issues = []
            if eval_result.is_missing_sentences: issues.append("원문의 핵심 문장 누락")
            if eval_result.is_grammar_broken: issues.append("치명적인 한국어 문법 붕괴")
            feedback = f"번역에 다음의 치명적 문제가 있습니다: [{', '.join(issues)}]. 단어 선택은 건드리지 말고 빠진 문장이나 깨진 문법만 복구하세요."

        # 무한 루프 방지 (최대 3회까지만 재번역)
        if state.get("revision_count", 1) >= 3:
            print("   -> ⚠️ [강제 통과] 최대 재작성 횟수 초과.")
            return {"critic_feedback": "PASS"}

        if "PASS" in feedback.upper():
            print("   -> ✅ [통과] 번역 퀄리티가 좋습니다.")
            return {"critic_feedback": "PASS"}
        else:
            print(f"   -> ❌ [반려] {feedback}")
            return {"critic_feedback": feedback}
