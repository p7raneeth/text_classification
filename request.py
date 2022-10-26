

import requests

url = 'http://127.0.0.1:8080/inference'
myobj = {}

x = requests.post(url, json = myobj)

print(x.text)