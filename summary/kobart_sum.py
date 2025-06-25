import re
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import torch

class KoBARTSummarizer:
    def __init__(self, model_name="gogamza/kobart-base-v2"):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def summarize(self, text, max_length=100, min_length=30):
        # 줄바꿈 제거 + 첫 문장 제거
        def remove_first_sentence(text):
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return ' '.join(sentences[1:]) if len(sentences) > 1 else text

        cleaned_text = remove_first_sentence(text.replace("\n", " "))

        # 인코딩
        input_ids = self.tokenizer.encode(
            cleaned_text, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.device)

        # 요약 생성
        summary_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=5,
            repetition_penalty=2.5,
            no_repeat_ngram_size=4,
            length_penalty=1.2,
            early_stopping=True
        )

        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        # 불필요한 토큰 제거 (예: KB)
        output = re.sub(r"^(KB\s*)+", "", output)

        # 문장이 끊기지 않게 마지막 마침표까지 자름
        if "." in output:
            output = output[:output.rfind(".") + 1]

        return output
