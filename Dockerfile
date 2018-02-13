FROM python:3

WORKDIR /facenet
ADD . /facenet

RUN pip3 install -r ./requirements.txt
ENV PYTHONPATH /facenet/src:/facenet/src/models:/facenet/src/align

CMD ["python", "contributed/server.py"]
