from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import shutil
import uuid
from explanation import describe_image
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 인스턴스 생성
client = OpenAI(api_key=api_key)

app = FastAPI()

@app.post("/ask-image")
async def ask_image(file: UploadFile = File(...), question: str = Form(...)):
    image_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        # 이미지 저장
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 이미지 설명 생성
        image_description = describe_image(image_path)

        # 프롬프트 구성
        prompt = f"""
당신은 사람의 행동을 이미지 설명으로부터 분석하는 전문가입니다.

다음은 직원의 현재 모습이 담긴 이미지에 대한 설명입니다:
\"{image_description}\"

이 사람이 현재 **업무 중인지**, 아니면 **쉬고 있거나 딴짓을 하고 있는지** 판단해 주세요.

표정, 자세, 시선, 손 위치, 주변 물건(예: 모니터, 스마트폰, 음식, 의자에 기대기, 눕기 등)을 고려하여
▶ 업무 중인지 여부 (예: 업무 중, 쉬는 중, 집중 안함 등)
▶ 판단 근거를 구체적으로 설명해 주세요.

아래의 질문을 참고해 주세요:
\"{question}\"

정확하고 현실적인 관찰로, 한국어로 답변해 주세요.
"""

        # GPT-4 응답 생성
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()
        return JSONResponse(content={"description": image_description, "answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
