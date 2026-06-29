"""
이미지(차트, 표 등) 분석을 담당하는 Vision 모듈입니다.
"""
import os
import base64
from langchain_core.messages import HumanMessage

class VisionEngine:
    def __init__(self, llm):
        self.llm = llm
        
    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def analyze_image(self, image_path: str, query: str) -> str:
        if not image_path or not os.path.exists(image_path):
            return "[시스템 오류] 업로드된 이미지가 없거나 찾을 수 없습니다."
            
        print(f"\\n[Vision Engine] 실제 이미지 분석 실행: '{image_path}'")
        
        base64_image = self._encode_image_to_base64(image_path)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        )
        
        response = self.llm.invoke([message])
        return f"[Vision 분석 결과]\\n{response.content}"
