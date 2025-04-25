from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import base64 #이미지 바이너리 인코딩
from openai import OpenAI

# 환경변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

@app.post("/ask-image")
async def ask_image(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # 이미지 바이너리 → base64
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # 프롬프트 구성: 근무 판별 지시 포함
        system_instruction = """
너는 사람의 행동을 이미지로 분석하는 전문가야.

업로드된 이미지를 보고, 이 사람이 '업무 중'인지, 아니면 '쉬는 중' 또는 '딴짓 중'인지 판단해.
다음 조건을 종합적으로 고려해줘:
- 표정 (집중, 피로, 멍함 등)
- 시선 방향 (모니터 응시 여부)
- 자세 (앉아 있음, 눕거나 기대 있음)
- 손 위치 (키보드/마우스 위, 스마트폰 들고 있음 등)
- 주변 물건 (노트북, 스마트폰, 음식 등)

질문자는 이렇게 물어봤어:
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_instruction}\n\"{question}\" 에 답변해줘."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # GPT-4-Vision 호출
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1000
        )

        answer = response.choices[0].message.content.strip()
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
