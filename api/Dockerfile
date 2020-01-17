# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
FROM continuumio/miniconda3

RUN apt update

RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY . /app

WORKDIR /app

CMD ["python", "/app/app.py"]

