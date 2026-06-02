# SNUT 멋쟁이사자처럼 챗봇 AI

서울과학기술대학교 멋쟁이사자처럼 동아리 관련 질문에 답변하는 의도 분류(Intent Classification) 챗봇입니다.  
KLUE-BERT 모델을 파인튜닝하여 50가지 의도를 분류하고, FastAPI 서버로 제공합니다.

---

## 프로젝트 구조

```
.
├── app.py                  # FastAPI 서버 (추론 엔드포인트)
├── model.py                # KLUE-BERT 학습 스크립트
├── model.ipynb             # 학습 노트북
├── requirements.txt        # Python 의존성
├── Dockerfile              # Docker 이미지 빌드 설정
├── kluebert_intent_out/    # 학습된 모델 저장 디렉토리
├── qi_train/               # 학습 데이터 (CSV)
└── qi_test/                # 평가 데이터 (CSV)
```

---

## 지원 의도(Intent) 목록 (50가지)

| 카테고리 | 의도 |
|---|---|
| 동아리 기본 정보 | 동아리_소개, 동아리_역사, 동아리_분위기, 동아리_구성원, 동아리_네트워크 |
| 모집 / 지원 | 모집_기간, 모집_대상, 모집_인원, 모집_인재상, 지원_방법, 지원_자격, 선발_기준, 선발_일정 |
| 파트 안내 | 파트_AI, 파트_기획, 파트_디자인, 파트_백엔드, 파트_프론트엔드, 파트_선택 |
| 면접 팁 | 면접팁_공통, 면접팁_AI, 면접팁_기획, 면접팁_디자인, 면접팁_백엔드, 면접팁_프론트엔드 |
| 지원서 작성 팁 | 지원서_작성팁_공통, 지원서_작성팁_AI, 지원서_작성팁_기획, 지원서_작성팁_디자인, 지원서_작성팁_백엔드, 지원서_작성팁_프론트엔드 |
| 활동 / 운영 | 동아리_활동, 동아리_혜택, 활동_기간, 활동_빈도, 활동_시간부담, 활동_장소, 활동_성과, 운영_방식, 외부_활동, 졸업후_활동 |
| 커리큘럼 | 배우는_기술, 커리큘럼_형식, 세션_참여조건, 수료_조건, 초보자_가능 |
| 기타 | 회비_안내, 인사, 감사, 이스터에그 |

---

## 실행 방법

### 1. 모델 학습

```bash
pip install -r requirements.txt

python model.py
```

학습이 완료되면 `kluebert_intent_out/` 디렉토리에 모델이 저장됩니다.

### 2. 로컬 서버 실행

```bash
pip install fastapi uvicorn

uvicorn app:app --reload --port 8000
```

### 3. Docker로 실행

```bash
# 이미지 빌드
docker build -t snut-likelion-chatbot .

# 컨테이너 실행
docker run -p 8000:8000 snut-likelion-chatbot
```

---

## API 사용법

### `POST /chat`

**요청**
```json
{
  "text": "멋사에 지원하려면 어떻게 해야 하나요?"
}
```

**응답**
```json
{
  "response": "판단된 의도는 지원_방법입니다.",
  "matched_question": "지원_방법",
  "score": 0.9821
}
```

> `score`가 0.4 미만이면 `matched_question`이 `null`로 반환됩니다.

---

## 모델 정보

| 항목 | 내용 |
|---|---|
| 베이스 모델 | `klue/bert-base` |
| 학습 에폭 | 15 |
| 학습률 | 3e-5 |
| 배치 크기 | 16 |
| 최대 시퀀스 길이 | 32 |
| 평가 지표 | Accuracy, Macro F1 |

---

## 기술 스택

- **모델**: KLUE-BERT (`klue/bert-base`)
- **학습**: HuggingFace Transformers, PyTorch
- **서빙**: FastAPI, Uvicorn
- **컨테이너**: Docker
