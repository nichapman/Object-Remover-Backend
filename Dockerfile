FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y git
# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]
