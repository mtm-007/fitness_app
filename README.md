## Fitness Assistant

This is a Fitness assitant, a Rag implemented LLM Application


## Installing dependecies

`pipenv` is used for managing depedencies, and Python 3.12.  

Make sure you have pipenv installed:

```bash
pip install pipenv
```

### Installing dependecies

```bash
pipenv install
```

### Running Flask application

Running Flask application API
```bash
pipenv run python app.py
```

### Testing API
```bash

URL=http://127.0.0.1:5000
QUESTION="Can you explain how to do a Glute Bridge, I am not sure about the movement."

DATA='{
    "question" : "'${QUESTION}'"
    }'

curl -X POST  \
    -H "Content-Type: application/json" \
    -d "${DATA}" \
    ${URL}/question 

```

### Sending feedback

```bash
ID="33277a06-a8d0-4c6e-acba-c6b247bb058d" 

URL=http://127.0.0.1:5000

FEEDBACK_DATA='{
    "conversation_id" : "'${ID}'",
    "feedback" : 1
}'

curl -X POST  \
    -H "Content-Type: application/json" \
    -d "${FEEDBACK_DATA}" \
    ${URL}/feedback 
```
## Misc

```bash
pipenv run jupyter lab
```


## Ingestion

## Evaluation
For evaultion sytem, check the code in [notebooks/indexing_data_and_Evaluation.ipynb](notebooks/indexing_data_and_Evaluation.ipynb) notebook.

## Retrieval

Minsearch is used as a basic approach withput any boosting, gave the following metrics  

* hit_rate: 93.4% 
* MRR: 79.9% 

The improved search(with boosting)

* hit_rate: 95% 
* MRR: 87%

## RAG Flow

## Monitoring
