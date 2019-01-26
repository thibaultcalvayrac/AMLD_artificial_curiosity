FROM python:3.6-slim

RUN apt-get update && apt-get -q -y install --reinstall build-essential && apt-get -q -y install ffmpeg gcc

COPY . /app

RUN pip install --upgrade pip && pip install -r /app/requirements.txt && pip install gym[atari]

WORKDIR /app/src

ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888"]
