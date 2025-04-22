#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
from tqdm.auto import tqdm

import minsearch


load_dotenv()


client = OpenAI()


def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


prompt_template = """
You're a fitness instructor. Answer the QUESTION based on the CONTEXT from our exercises database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

entry_template = """ 
exercise_name : {exercise_name}
type_of_activity : {type_of_activity}
type_of_equipment : {type_of_equipment}
body_part : {body_part}
type : {type}
muscle_groups_activated : {muscle_groups_activated}
instructions : {instructions}
""".strip()
    
def build_prompt(query, search_results):
    
    context = ""
    
    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


query =  "Can you explain how to do a Glute Bridge, I'm not sure about the movement."


search_results = search(query)
prompt = build_prompt(query, search_results)


def llm(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def rag(query, model="gpt-4o-mini",):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer



answer = rag("What specific muscle groups are predominantly activated during the Cable Face Pull exercise?")
print(answer)


# ### Retrieval Evaluation

df_questions = pd.read_csv('../data/ground_truth_retrieval.csv')


df_questions.head()


ground_truth = df_questions.to_dict(orient="records")

ground_truth[0]


def hit_rate(relevance_input):
    cnt = 0
    for line in relevance_input:
        if True in line:
            cnt = cnt + 1
            
    return cnt / len(relevance_input)


def mrr(relavance_input):
    total_score = 0

    for line in relavance_input:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relavance_input)



def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# #### Finding the best parameters

df_Validation = df_questions[:100]


import random

def simple_optimize(param_ranges, objective_function, n_iterations =10):
    best_params = None
    best_score = float('-inf')

    for _ in range(n_iterations):
        #Generate random parameters
        current_params = {}

        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)
        #Evaluate objective function
        current_score = objective_function(current_params)


        #update best if current is better
        if current_score > best_score: #change to > if maximizing
            best_score = current_score
            best_params = current_params

    return best_params, best_score


gt_val = df_Validation.to_dict(orient="records")


def minsearch_search(query, boost= None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


param_ranges = {
    'exercise_name': (0.0, 3.0),
     'type_of_activity': (0.0, 3.0),
     'type_of_equipment': (0.0, 3.0),
     'body_part': (0.0, 3.0),
     'type': (0.0, 3.0),
     'muscle_groups_activated': (0.0, 3.0),
     'instructions': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)
                                
    results = evaluate(gt_val, search_function)
    #return results['mrr']
    return results['hit_rate']


simple_optimize(param_ranges, objective, n_iterations =10)


def minsearch_improved(query):
    boost = {
    
    'exercise_name': 2.74,
    'type_of_activity': 0.45,
    'type_of_equipment': 1.06,
    'body_part': 0.094,
    'type': 2.40,
    'muscle_groups_activated': 0.58,
    'instructions': 1.66
    }


    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))



{'hit_rate': 0.946923076923077, 'mrr': 0.868453296703297}


# ### Rag Evaluation
# #### LLM As a Judge

prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


record = ground_truth[0]
question = record['question']
answer_llm = rag(question)


print(answer_llm)


prompt = prompt2_template.format(question = question, answer_llm= answer_llm)
print(prompt)


llm(prompt)


df_sample = df_questions.sample(n=200, random_state =1)


sample = df_sample.to_dict(orient="records")


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question)

    prompt = prompt2_template.format(
        question = question, 
        answer_llm= answer_llm
    )
    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


df_eval = pd.DataFrame(evaluations, columns = ['record', 'answer', 'evaluation'])



df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])



del df_eval['record']
del df_eval['evaluation']





evaluations_gpt_4_1 = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question, model = 'gpt-4.1-mini')

    prompt = prompt2_template.format(
        question = question, 
        answer_llm= answer_llm
    )
    evaluation = llm(prompt, model = 'gpt-4.1-mini')
    evaluation = json.loads(evaluation)

    evaluations_gpt_4_1.append((record, answer_llm, evaluation))


df_eval_2 = pd.DataFrame(evaluations_gpt_4_1, columns = ['record', 'answer', 'evaluation'])



df_eval_2['id'] = df_eval_2.record.apply(lambda d: d['id'])
df_eval_2['question'] = df_eval_2.record.apply(lambda d: d['question'])

df_eval_2['relevance'] = df_eval_2.evaluation.apply(lambda d: d['Relevance'])
df_eval_2['explanation'] = df_eval_2.evaluation.apply(lambda d: d['Explanation'])


del df_eval_2['record']
del df_eval_2['evaluation']


df_eval_2.relevance.value_counts(normalize=True)

