FROM python:3.10-slim

WORKDIR /

COPY /requirements.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]