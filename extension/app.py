from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# === KoGPT 모델로 교체 ===
MODEL_NAME = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def kogpt_expand(user_input, max_new_tokens):
    if not user_input.strip():
        return "⚠️ 빈 입력입니다."
    prompt = f"[입력]\n{user_input}\n\n[출력]\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=True,  # 확률적 생성 (샘플링)
        top_p=0.92,  # nucleus sampling(다양성)
        top_k=50,  # top-k 샘플링도 병행
        temperature=0.7,  # 분포 평탄화
        num_beams=4,  # 빔서치(다양한 후보)
        no_repeat_ngram_size=3,  # 3그램 반복 금지
        repetition_penalty=1.2,  # 반복 패널티
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # [출력] 이후만 추출
    if "[출력]" in decoded:
        result = decoded.split("[출력]")[-1].strip()
    else:
        result = decoded.strip()
    return result


import re


def clean_output(text):
    # 0) 꺽쇠(<, >)와 대괄호([, ])를 만나면 그 지점에서 잘라 버리기
    #    < 혹은 [ 가 나오면 그 앞부분만 취함 -> 여기서 () 이것도 포함
    text = re.split(r"[<\[\]\(\)]", text)[0]

    # 1) 줄바꿈을 공백으로 바꿔 한 줄로 이어붙이기
    text = text.replace("\n", " ")

    # 2) 연속 공백은 하나로
    text = re.sub(r"\s+", " ", text).strip()

    # 3) 마지막 온전한 문장까지만 남기기 (마지막 마침표 기준)
    last_dot = text.rfind(".")
    if last_dot != -1:
        text = text[: last_dot + 1]

    # 4) 마지막 온전한 문장까지만 남기기 (마지막 물결표 기준)
    last_dot = text.rfind("~")
    if last_dot != -1:
        text = text[: last_dot + 1]

    return text


@app.route("/")
def home():
    return "Hello BlogGen-KoGPT! Send POST to /expand with 'text'."


@app.route("/expand", methods=["POST"])
def expand():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"expanded_text": "⚠️ 빈 입력입니다."})

    # 1) user_input만 토큰화해서 길이 측정
    user_inputs = tokenizer(text, return_tensors="pt")
    input_len = user_inputs["input_ids"].shape[1]

    # 2) 그대로 max_new_tokens에 할당
    max_new_tokens = input_len * 2

    print(f"[입력 텍스트] {text}  → 토큰 수: {input_len}")
    sentence = kogpt_expand(text, max_new_tokens=max_new_tokens)
    result = clean_output(sentence)
    return jsonify({"expanded_text": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
