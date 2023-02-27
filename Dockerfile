FROM python:3.10
WORKDIR /
ADD . .
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python3", "./rlagent.py"]