FROM python:3.10-slim
WORKDIR /home/agent
COPY . .
RUN apt-get update && apt-get install -y build-essential libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["bash","start.sh"]
