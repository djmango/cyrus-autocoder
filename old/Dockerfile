FROM tensorflow/tensorflow:2.5.0-gpu

WORKDIR /darius-trainer
ADD requirements.txt ./requirements.txt
# RUN apt-get update && apt-get install -y ca-certificates
RUN pip install --no-cache-dir -r /darius-trainer/requirements.txt
CMD [ "python", "/darius-trainer/train.py"]
# CMD [ "python", "/darius-trainer/predict.py"]