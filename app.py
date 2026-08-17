from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "clf.joblib"))

SCORE_THRESHOLD = 0.4


class ChatRequest(BaseModel):
    text: str


@app.post("/chat")
async def chat(request: ChatRequest):
    X = vectorizer.transform([request.text])
    proba = clf.predict_proba(X)[0]
    idx = proba.argmax()
    intent, score = clf.classes_[idx], round(float(proba[idx]), 4)

    # 답변 텍스트는 이 서비스 책임이 아님 — intent만 반환, 실제 답변은 다운스트림(백엔드)에서 매핑
    return {
        "intent": intent if score > SCORE_THRESHOLD else None,
        "score": score,
    }
