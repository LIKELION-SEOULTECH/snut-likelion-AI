from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

# 1. 검색용 SBERT 로드
sbert = SentenceTransformer("./sbert_finetuned")

# 2. 생성용 fine-tuned KoGPT 로드
tokenizer = AutoTokenizer.from_pretrained("./kogpt_finetuned")
generator = AutoModelForCausalLM.from_pretrained("./kogpt_finetuned", device_map="auto")

# 3. 데이터 로딩 및 질문 임베딩
df = pd.read_excel("chatbot_augmented.xlsx")
questions = df["질문"].tolist()
answers = df["답변"].tolist()
question_embeddings = sbert.encode(questions, convert_to_tensor=True)


# 4. 유사 질문 검색 함수
def find_similar(user_input, top_k=1):
    input_embedding = sbert.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, question_embeddings)[0]
    top_result = torch.topk(cos_scores, k=top_k)
    best_idx = top_result.indices[0].item()
    score = top_result.values[0].item()
    return questions[best_idx], answers[best_idx], score


# 5. 생성 함수
def generate_answer(user_input, matched_question, matched_answer):
    prompt = f"""당신은 서울과학기술대학교 멋쟁이사자처럼 동아리의 공식 안내 챗봇입니다.  
사용자의 질문에 대해 아래에 제공된 기존 답변을 참고하여, 동일한 내용을 자연스럽고 친절한 말투로 전달해 주세요.  
사실에 기반한 응답만 제공하며, 새로운 정보를 임의로 추가하지 마세요. 

[사용자 질문]
{user_input}

[기존 답변]
{matched_answer}

[응답]
"""
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"].to(generator.device)
    attention_mask = inputs["attention_mask"].to(generator.device)

    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 🟡 후처리: 잘못된 띄어쓰기 고치기
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = raw_output.split("[응답]")[-1].strip()

    # 🔧 띄어쓰기 자동 수정
    fixes = {
        "멋사 은": "멋사은",
        "멋쟁이사자처럼 은": "멋쟁이사자처럼은",
        "멋사 는": "멋사는",
        "멋사 의": "멋사의",
    }
    for wrong, correct in fixes.items():
        response = response.replace(wrong, correct)

    return response


# 6. 통합 챗봇 함수
def hybrid_chatbot(user_input, fallback_threshold=0.5):
    matched_q, matched_a, score = find_similar(user_input)
    if score < fallback_threshold:
        return "정확한 정보를 찾기 어려워요. 인스타 DM으로 문의 주세요 👉 instagram.com/likelion_st"
    return generate_answer(user_input, matched_q, matched_a)


# 7. 테스트
if __name__ == "__main__":
    test_questions = [
        "세션은 꼭 참여해야 하나요?",
        "해커톤은 꼭 나가야 돼?",
        "중앙 OT는 필수인가요?",
        "세션 빠지면 수료 못하나요?",
        "동아리 활동 시간은 어떻게 되나요?",
        "스터디는 의무인가요?",
    ]

    for q in test_questions:
        print("👤 사용자 질문:", q)
        matched_q, matched_a, score = find_similar(q)
        print("🔍 가장 유사한 질문:", matched_q)
        print("📌 기존 답변:", matched_a)
        print("🤖 생성된 답변:")
        print(generate_answer(q, matched_q, matched_a))
        print("-" * 60)
