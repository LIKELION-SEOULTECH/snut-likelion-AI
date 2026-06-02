# =========================================================
# train_eval_kluebert_intent.py — 학습 + 평가 + 저장 통합 안정판
# (GPU 사용 / Windows 호환성 / 멀티프로세싱 오류 수정)
# =========================================================
import os
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import font_manager, rc
import platform  # OS 확인을 위해 추가


# ---------------------------------------------------------
# 0️⃣ 한글 폰트 설정 (Windows/Linux 자동 호환)
# ---------------------------------------------------------
def set_korean_font():
    """OS에 맞춰 한글 폰트 설정"""
    system_name = platform.system()

    if system_name == "Windows":
        font_name = font_manager.FontProperties(
            fname="c:/Windows/Fonts/malgun.ttf"
        ).get_name()
        rc("font", family=font_name)
    elif system_name == "Linux":
        # 사용자가 제공한 경로
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(font_path):
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc("font", family=font_name)
        else:
            print(
                "Linux에서 NanumGothic 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다."
            )
    else:
        print(
            f"{system_name} OS는 폰트 설정을 지원하지 않습니다. 기본 폰트를 사용합니다."
        )

    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지
    print(f"✅ Font setup for {system_name} complete.")


# ---------------------------------------------------------
# 2️⃣ 데이터 로드 (함수 정의)
# ---------------------------------------------------------
def load_csvs(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(
                    os.path.join(folder, f), encoding="utf-8", on_bad_lines="skip"
                )
                # "question " (공백 포함) 컬럼명 처리
                if "question " in df.columns:
                    df = df.rename(columns={"question ": "question"})

                if "question" in df.columns and "intent" in df.columns:
                    dfs.append(df[["question", "intent"]])
                else:
                    print(
                        f"Warning: {f} 파일에 'question' 또는 'intent' 컬럼이 없습니다."
                    )

            except Exception as e:
                print(f"Error loading {f}: {e}")

    if not dfs:
        raise ValueError(f"폴더에 유효한 CSV 파일이 없습니다: {folder}")

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------
# 7️⃣ 지표 정의 (함수 정의)
# ---------------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------
# ⬇️ 메인 실행 로직 (if __name__ == '__main__': 내부에서 실행)
# ---------------------------------------------------------
def main():

    # 폰트 설정 실행
    set_korean_font()

    # 1️⃣ 경로 설정
    train_dir = "./qi_train"
    test_dir = "./qi_test"
    output_dir = "./kluebert_intent_out"

    # GPU/CPU 장치 설정 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ---------------------------------------------------------
    # 2️⃣ 데이터 로드 (실행)
    # ---------------------------------------------------------
    train_df = load_csvs(train_dir).dropna()
    test_df = load_csvs(test_dir).dropna()

    print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ---------------------------------------------------------
    # 3️⃣ 라벨 인코딩
    # ---------------------------------------------------------
    le = LabelEncoder()
    all_labels = sorted(list(set(train_df["intent"]) | set(test_df["intent"])))
    le.fit(all_labels)
    train_df["label"] = le.transform(train_df["intent"])
    test_df["label"] = le.transform(test_df["intent"])

    num_labels = len(le.classes_)
    print(f"✅ {num_labels}개의 라벨 발견: {list(le.classes_)}")

    # ---------------------------------------------------------
    # 4️⃣ Dataset 변환
    # ---------------------------------------------------------
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    # ---------------------------------------------------------
    # 5️⃣ Tokenizer
    # ---------------------------------------------------------
    model_name = "klue/bert-base"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(
            batch["question"], truncation=True, padding="max_length", max_length=32
        )

    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    # ---------------------------------------------------------
    # 6️⃣ 모델 정의
    # ---------------------------------------------------------
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    # model.to(device) # Trainer가 자동으로 처리해줍니다.

    # ---------------------------------------------------------
    # 8️⃣ 학습 설정
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_eval=True,
        eval_strategy="epoch",  # ⬅️ evaluation_strategy에서 eval_strategy로 변경
        save_strategy="epoch",
        num_train_epochs=15,          
        learning_rate=3e-5,           
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        weight_decay=0.01,            
        load_best_model_at_end=True,  
        metric_for_best_model="f1",   
    )

    # ---------------------------------------------------------
    # 9️⃣ Trainer
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,  # ✅ 'tokenizer' 대신 'processing_class' 사용
        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------------
    # 🔟 학습 및 평가
    # ---------------------------------------------------------
    print("🚀 KLUE-BERT 모델 학습을 시작합니다...")
    trainer.train()
    eval_result = trainer.evaluate()
    print("\n📊 Evaluation Result:", eval_result)

    # ---------------------------------------------------------
    # 11️⃣ 모델 저장
    # ---------------------------------------------------------
    print(f"✅ 모델 및 토크나이저를 {output_dir}에 저장합니다.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 라벨 맵 저장
    label_map = {i: label for i, label in enumerate(le.classes_)}
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # LabelEncoder 저장 (torch.save 대신 pickle 권장)
    import pickle

    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # ---------------------------------------------------------
    # 12️⃣ Confusion Matrix 시각화
    # ---------------------------------------------------------
    print("📊 Confusion Matrix를 생성합니다...")
    predictions = trainer.predict(test_ds)
    y_true = test_df["intent"].tolist()
    y_pred_ids = np.argmax(predictions.predictions, axis=1)
    y_pred = [le.classes_[i] for i in y_pred_ids]

    cm = confusion_matrix(y_true, y_pred, labels=list(le.classes_))

    # 라벨이 너무 많으면(예: 30개 초과) annot=False로 변경
    show_annotations = len(le.classes_) <= 30

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=show_annotations,
        fmt="d",  # 정수로 표현
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("KLUE-BERT Intent Classification Confusion Matrix", fontsize=14)
    plt.tight_layout()

    # 그래프를 파일로 저장
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"✅ Confusion Matrix가 {output_dir}/confusion_matrix.png 에 저장되었습니다.")

    # (선택) plt.show() # 로컬 환경에서 바로 보려면 주석 해제

    print("\n✅ Training + Evaluation Completed Successfully.")


# ---------------------------------------------------------
# 🚀 스크립트 실행 지점
# (Windows에서 멀티프로세싱 오류를 방지하기 위해 필수)
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
