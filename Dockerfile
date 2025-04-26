FROM python:3.12-slim

WORKDIR /app

RUN pip install pipenv

COPY data/data.csv data/data.csv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy -ignore-pipenv

COPY fitness_assistant .

CMD ["python", "app.py"]