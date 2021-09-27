FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /darius-trainer
ADD requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r /darius-trainer/requirements.txt
CMD [ "python", "/darius-trainer/train.py"]
# CMD [ "python", "/darius-trainer/predict.py"]