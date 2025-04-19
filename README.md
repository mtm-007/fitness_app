## Fitness Assistant

This is a Fitness assitant, a Rag implemented LLM Application


## Running it

`pipenv` is used for managing depedencies, and Python 3.12.  

Make sure you have pipenv installed:

```bash
pip install pipenv
```

## Installing dependecies

```bash
pipenv install
```

## Running Jupyter for dev experiments:

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

* hit_rate: 94.61% 
* MRR: 86.52%

## RAG Flow

## Monitoring
