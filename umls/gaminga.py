import requests
from pathlib import Path
import json
import random
import re

HERE = Path(__file__).parent

# randomly choose an api key from key file
# api_key = input('insert api key: ')
with open(HERE.joinpath('keys.json')) as keysfile:
    api_keys = json.load(keysfile)

if random.randint(0, 1) == 0:
    api_key = api_keys['umls_al']
else:
    api_key = api_keys['umls_sulaiman']

headers = {'content-type': 'application/x-www-form-urlencoded'}

# get tgt via api key
r = requests.post('https://utslogin.nlm.nih.gov/cas/v1/api-key', params={'apikey': api_key}, headers=headers)

if r.status_code == 201:
    print('API Key 201 success!')
else:
    print('API Key failure.. ' + str(r.status_code))
    exit()

# pull tgt url
tokenurl = re.findall(r'action="([^"]+)"', r.text)[0]
print(tokenurl)

# use tgt to get service ticket
r = requests.post(tokenurl, params={'service': 'http://umlsks.nlm.nih.gov'}, headers=headers)

if r.status_code == 200:
    print('\nSingle-use Token 200 success!')
    print(r.text)
else:
    print('Single-use token failure.. ' + str(r.status_code))
    exit()


