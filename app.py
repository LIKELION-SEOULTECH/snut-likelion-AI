from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

app = Flask(__name__)

# 모델 및 데이터 로드
sbert = SentenceTransformer("./sbert_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./kogpt_finetuned")
generator = AutoModelForCausalLM.from_pretrained("./kogpt_finetuned", device_map="auto")

df = pd.read_excel("chatbot_augmented.xlsx")
questions = df["질문"].tolist()
answers = df["답변"].tolist()
question_embeddings = sbert.encode(questions, convert_to_tensor=True)


def find_similar(user_input, top_k=1):
    input_embedding = sbert.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, question_embeddings)[0]
    top_result = torch.topk(cos_scores, k=top_k)
    best_idx = top_result.indices[0].item()
    score = top_result.values[0].item()
    return questions[best_idx], answers[best_idx], score


def generate_answer(user_input, matched_question, matched_answer):
    prompt = f"""당신은 서울과학기술대학교 멋사 동아리의 공식 안내 챗봇입니다.

[사용자 질문]에 대해, 아래 [기존 답변]을 참고하여  
👉 **질문에 직접적으로 관련 있는 내용만**  
👉 **2~3문장 이내로 짧고 간결하게**  
👉 **중복 없이 요약해서 전달**하세요.  
💡 새로운 내용을 추가하지 마세요.

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
        max_new_tokens=60,
        do_sample=True,
        temperature=0.3,
        top_p=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    return (
        tokenizer.decode(output[0], skip_special_tokens=True)
        .split("[응답]")[-1]
        .strip()
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data["text"]
    matched_q, matched_a, score = find_similar(user_input)

    if score < 0.5:
        return jsonify(
            {
                "response": "정확한 정보를 찾기 어려워요. 인스타 DM으로 문의 주세요 👉 instagram.com/likelion_st",
                "matched_question": None,
                "score": float(score),
            }
        )

    answer = generate_answer(user_input, matched_q, matched_a)
    return jsonify(
        {"response": answer, "matched_question": matched_q, "score": float(score)}
    )


@app.route("/")
def home():
    return "멋사 챗봇 서버입니다. POST /chat 으로 질문을 보내세요."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
