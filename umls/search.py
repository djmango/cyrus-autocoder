import requests
from pathlib import Path
import json
import random
import re
import authhandler


# https://documentation.uts.nlm.nih.gov/rest/home.html API

# config
BASE_URL = 'https://uts-ws.nlm.nih.gov/rest'
authy = authhandler.UMLS_handler()
CUI = 'C0155502'
# headers = {'content-type': 'application/x-www-form-urlencoded'}
# healthcheck_url = 'https://hc-ping.com/db5f3705-2427-4f9e-b104-163d9010c7bd'

# # log that the script has been run
# try:
#     requests.get(healthcheck_url + "/start", timeout=5)
# except requests.exceptions.RequestException:
#     # If the network request fails for any reason, we don't want
#     # it to prevent the main job from running
#     pass

# r = requests.get(healthcheck_url)
# print(r.text)

# get ticket

def definition(searchstring):
    ticket = authy.getSingleUseTicket()
    search_url = f'{BASE_URL}/content/current/CUI/{CUI}/definitions'

    params = {
        'ticket': ticket,
        'CUI' : CUI,
    }

    x = requests.get(search_url, params=params, headers=authhandler.UMLS_handler.HEADERS)

    defdata = x.text
    print(defdata)
    return defdata

def relation(searchstring):
    ticket = authy.getSingleUseTicket()
    search_url = f'{BASE_URL}/content/current/CUI/{CUI}/relations'

    params = {
        'ticket': ticket,
        'CUI' : CUI,
    }

    c = requests.get(search_url, params=params, headers=authhandler.UMLS_handler.HEADERS)

    refdata = c.text
    print(refdata)
    return refdata


def search(searchstring):
    ticket = authy.getSingleUseTicket()
    search_url = f'{BASE_URL}/search/current'

    params = {
        'string': searchstring,
        'ticket': ticket,
        'inputType' : 'code'
    }

    r = requests.get(search_url, params=params, headers=authhandler.UMLS_handler.HEADERS)

    healthdata = r.text
    print(healthdata)
    return healthdata

# done!

if __name__ == '__main__':
    search('Kaschin-Beck disease, right shoulder')