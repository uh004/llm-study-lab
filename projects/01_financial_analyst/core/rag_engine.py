"""
문서(PDF/텍스트) 검색을 담당하는 RAG 엔진 모듈입니다.
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        
    def load_and_index_pdf(self, pdf_path: str):
        """PDF를 읽어 벡터 데이터베이스(Chroma)에 인덱싱합니다."""
        print(f"\\n[RAG Engine] PDF 로딩 시작: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # 메모리 상의 임시 벡터 스토어 생성
        self.vector_store = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        print(f"[RAG Engine] {len(splits)}개의 청크로 분할하여 벡터 스토어 생성 완료.")
        
    def search(self, query: str) -> str:
        if self.vector_store is None:
            return "[시스템 오류] 업로드된 문서가 없거나 인덱싱되지 않았습니다."
            
        print(f"[RAG Engine] 실제 벡터 검색 실행: '{query}'")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke(query)
        
        context = "\\n".join([doc.page_content for doc in results])
        return f"[PDF 검색 결과]\\n{context}"
