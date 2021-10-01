import requests
import logging
import random
import re
import json
from authhandler import UMLS_handler

print("What is your CUI Code?")

class Person:
    e = 'cui'
    def __init__(CUI, ID):
        CUI.ID = ID
    

class UMLS_cui:
    BASE_URL = 'https://uts-ws.nlm.nih.gov/rest'
    
    def __init__(self, cui, authhandlerInstance):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not re.match(r'^C\d{3,10}$', cui):
            raise ValueError(f'{cui} is not a a vlai')
        self.cui = cui
        self.authy = authhandlerInstance

    def ensureResult(self, request):
        if request.status_code == 200:
            self.logger.debug('Request successful: HTTP ' + str(request.status_code))
            return request
        else:
            raise RuntimeError(f'Request failure: HTTP {str(request.status_code)} {request.text}')

    def relations(self):
        request_url = f'{UMLS_cui.BASE_URL}/content/current/CUI/{self.cui}/relations'
        request = requests.get(request_url, params={'ticket': self.authy.getSingleUseTicket()}, headers=UMLS_handler.HEADERS)

        return json.loads(self.ensureResult(request).text)
    
    def definition(self):
        request_url = f'{UMLS_cui.BASE_URL}/content/current/CUI/{self.cui}/definitions'

        x = requests.get(request_url, params={'ticket': self.authy.getSingleUseTicket()}, headers=UMLS_handler.HEADERS)

        return json.loads(self.ensureResult(x).text)


if __name__ == '__main__':
    # p1 = Person(input("Enter CUI: "))
    # print(p1.ID)
    ourAuthy = UMLS_handler()

    cui = input('Enter CUI: ')
    joebopp = UMLS_cui(cui, ourAuthy)

    result = joebopp.definition()
    print(result)