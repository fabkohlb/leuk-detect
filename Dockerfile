FROM python:3.10.6-bullseye

COPY requirements.txt requirements.txt

RUN apt-get clean -y
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ml_logic ml_logic
COPY api api
COPY models models

EXPOSE 8080

CMD uvicorn api.simple_api:app --host 0.0.0.0 --port $PORT
