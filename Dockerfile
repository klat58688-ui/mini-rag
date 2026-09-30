FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -i https://pypi.org/simple -r requirements.txt

COPY . .

EXPOSE 8000

# 默认启动 API；demo 时也可以 `docker run ... python -m src.cli "..."`
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
