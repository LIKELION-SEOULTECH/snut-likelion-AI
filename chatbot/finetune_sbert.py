import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    util,
)
from torch.utils.data import DataLoader
from itertools import combinations

# 1. 데이터 로딩
df = pd.read_excel("E:/star/BabyLION/chatbot/chatbot_augmented.xlsx")
df = df[["질문", "답변"]].dropna()

# 2. 같은 답변을 공유하는 질문 그룹에서 유사 쌍 생성
qa_pairs = []
for answer, group in df.groupby("답변"):
    questions = group["질문"].tolist()
    for q1, q2 in combinations(questions, 2):
        qa_pairs.append((q1, q2))

# 3. InputExample로 변환
train_examples = [InputExample(texts=[q1, q2]) for q1, q2 in qa_pairs]

# 4. 디바이스 설정 (CUDA 사용 가능 시 CUDA로)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 사용 디바이스:", device)

# 5. 모델 로딩 및 디바이스 이동
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
model = model.to(device)

# 6. DataLoader 및 Loss 구성
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 7. 평가자 설정 (일부 샘플만 사용)
eval_examples = train_examples[:20]
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples, name="sbert-eval"
)

# 8. 파인튜닝
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True,
)

# 9. 모델 저장 (명시적)
model.save("./sbert_finetuned")
print("✅ SBERT 파인튜닝 완료 및 저장됨: ./sbert_finetuned")

print("✅ SBERT 파인튜닝 완료 및 저장됨: ./sbert_finetuned")

# 10. 유사도 테스트
if __name__ == "__main__":
    print("\n📐 유사도 테스트 예제")

    model = SentenceTransformer("./sbert_finetuned")
    model = model.to(device)

    print("현재 모델이 로드된 디바이스:", model.device)
    print("CUDA 사용 가능 여부:", torch.cuda.is_available())

    query1 = "세션 꼭 나가야 해요?"
    query2 = "정기 세션 불참하면 안 되나요?"

    embeddings = model.encode([query1, query2], convert_to_tensor=True, device=device)
    score = util.cos_sim(embeddings[0], embeddings[1])
    print(f"'{query1}' ↔ '{query2}' 유사도 점수:", round(score.item(), 4))
