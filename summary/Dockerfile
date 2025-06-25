# 1. Python 기반 이미지
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 파일 복사
COPY requirements.txt ./
COPY app.py ./
COPY kobart_sum.py ./

# 4. 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 포트 설정
EXPOSE 8000

# 6. 실행 명령어
CMD ["python", "app.py"]
