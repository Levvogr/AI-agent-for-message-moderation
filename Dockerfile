FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential python3-pip cmake

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]