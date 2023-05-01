#FROM ubuntu:latest
FROM python:3.9-buster
LABEL authors="aarslan"

COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR "/app"

RUN pip install -r /tmp/requirements.txt

EXPOSE 8050
ENTRYPOINT [ "python3", "text_processor_dash.py" ]