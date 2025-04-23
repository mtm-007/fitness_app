import requests
from urllib3 import response

url = "http://127.0.0.1:5000/question"
question = "Can you explain how to do a Glute Bridge, I am not sure about the movement."
data= {"question" : question}

response = requests.post(url, json=data)

print(response.json())