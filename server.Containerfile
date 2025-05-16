FROM docker.io/python:3.13-slim
    MAINTAINER Linus Kirkwood <linuskirkwood@gmail.com>

RUN apt-get update && apt-get install -y caddy

RUN mkdir /opt/knightvision-server
# COPY pyproject.toml /opt/knightvision-server/pyproject.toml

WORKDIR /opt/knightvision-server
RUN pip install ultralytics==8.3.113
RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install opencv-python pillow numpy==1.26.4 pyyaml

COPY src /opt/knightvision-server/src
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
