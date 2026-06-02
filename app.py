from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI()

# 1. 라벨 리스트 (LABEL_44를 "활동_기간"으로 바꾸기 위함)
INTENT_LABELS = [
    '감사', '동아리_구성원', '동아리_네트워크', '동아리_분위기', '동아리_소개', 
    '동아리_역사', '동아리_혜택', '동아리_활동', '면접팁_AI', '면접팁_공통', 
    '면접팁_기획', '면접팁_디자인', '면접팁_백엔드', '면접팁_프론트엔드', '모집_기간', 
    '모집_대상', '모집_인원', '모집_인재상', '배우는_기술', '선발_기준', 
    '선발_일정', '세션_참여조건', '수료_조건', '외부_활동', '운영_방식', 
    '이스터에그', '인사', '졸업후_활동', '지원_방법', '지원_자격', 
    '지원서_작성팁_AI', '지원서_작성팁_공통', '지원서_작성팁_기획', '지원서_작성팁_디자인', '지원서_작성팁_백엔드', 
    '지원서_작성팁_프론트엔드', '초보자_가능', '커리큘럼_형식', '파트_AI', '파트_기획', 
    '파트_디자인', '파트_백엔드', '파트_선택', '파트_프론트엔드', '활동_기간', 
    '활동_빈до', '활동_빈도', '활동_성과', '활동_시간부담', '활동_장소', '회비_안내'
]

# 2. POST 요청 시 받을 데이터 구조 정의
class ChatRequest(BaseModel):
    text: str

MODEL_PATH = "./kluebert_intent_out"
classifier = pipeline("text-classification", model=MODEL_PATH, device=-1)

@app.post("/chat")  # GET /predict 대신 POST /chat 사용
async def chat(request: ChatRequest):
    # 3. 예측 실행
    result = classifier(request.text)[0]
    
    # 4. LABEL_XX 에서 숫자만 뽑아서 라벨 이름으로 변환
    label_idx = int(result['label'].split('_')[1])
    intent_name = INTENT_LABELS[label_idx]
    score = round(result['score'], 4)

    # 5. 명세서 형식에 맞춘 리턴 (점수가 낮으면 null 처리 예시)
    return {
        "response": f"판단된 의도는 {intent_name}입니다.", # 여기에 답변 로직 추가 가능
        "matched_question": intent_name if score > 0.4 else None,
        "score": score
    }