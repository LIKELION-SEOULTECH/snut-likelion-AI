import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import evaluate
from transformers import EvalPrediction

# 1. 데이터 로드
df = pd.read_excel("E:/star/BabyLION/chatbot/chatbot_augmented.xlsx")
df = df[["질문", "답변"]].dropna()
df["text"] = df.apply(lambda row: f"질문: {row['질문']}\n답변: {row['답변']}", axis=1)
dataset = Dataset.from_pandas(df[["text"]])

# 2. 모델 및 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))


# 3. 토큰화
def tokenize(example):
    tokens = tokenizer(
        example["text"], max_length=512, padding="max_length", truncation=True
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. 평가 지표
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    preds[preds == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_score = bleu.compute(
        predictions=decoded_preds, references=[[l] for l in decoded_labels]
    )
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu_score["bleu"], "rougeL": rouge_score["rougeL"]}


# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./kogpt_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    prediction_loss_only=True,  # ✅ logits 안 저장해서 메모리 폭발 안 함
    save_steps=0,
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

eval_dataset = tokenized_dataset.select(range(50))
# 6. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. 학습 실행
trainer.train()


# 8. 저장
model.save_pretrained("./kogpt_finetuned")
tokenizer.save_pretrained("./kogpt_finetuned")

print("✅ KoGPT 파인튜닝 완료 및 저장됨: ./kogpt_finetuned")
