FROM python:3.9-slim

RUN apt-get update && apt-get install -y git build-essential

WORKDIR '/snut/app'

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py .
COPY ./sbert_finetuned .

EXPOSE 8000

CMD ["python", "app.py"]
