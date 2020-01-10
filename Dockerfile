FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt update && \
    apt install -y libsm6 libxext6 libxrender-dev

WORKDIR /tf/src

# Only need the base requirements (i.e. excluding Tensorflow)
COPY requirements_base.txt .
RUN pip install -r requirements_base.txt
