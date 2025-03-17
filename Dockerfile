FROM python:3.10.6-buster

COPY ml_logic /ml_logic
COPY api /api
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

CMD uvicorn api.simple_api:app --host 0.0.0.0 --port $PORT
