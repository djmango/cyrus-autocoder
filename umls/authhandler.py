import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

import requests

class UMLS_handler():
    """
    class to handle tickets and authentication requests with UMLS API
    """
    HEADERS = {'content-type': 'application/x-www-form-urlencoded'}
    HERE = Path(__file__).parent

    def __init__(self, api_key_index=random.randint(0, 1)):
        # init our instance logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # randomly choose an api key from key file
        with open(UMLS_handler.HERE.joinpath('keys.json')) as keysfile:
            api_keys = json.load(keysfile)

        if api_key_index == 0:
            self.api_key = api_keys['umls_al']
        else:
            self.api_key = api_keys['umls_sulaiman']
        
        self.requestTGT()

    def requestTGT(self):
        """ get a new TGT via api key
        Raises:
            RuntimeError: Failed getting TGT

        Returns:
            str: TGT in a url
        """

        self.lastTGTtime = datetime.now()
        r = requests.post('https://utslogin.nlm.nih.gov/cas/v1/api-key', params={'apikey': self.api_key}, headers=UMLS_handler.HEADERS)

        if r.status_code == 201:
            self.logger.info('Successfully got TGT: HTTP ' + str(r.status_code))
        else:
            raise RuntimeError('Failed getting TGT: HTTP ' + str(r.status_code))

        # pull tgt url
        self.tgturl = re.findall(r'action="([^"]+)"', r.text)[0]
        return self.tgturl

    def ensureTGT(self):
        """ check if 8 hours have passed since we got the token, if so then request a new one

        Returns:
            tuple: (gotToken, token) first item in tuple is if we got a new TGT, the second item is the TGT
        """
        if not hasattr(self, 'lastTGTtime') or (datetime.now() - self.lastTGTtime).total_seconds() >= 28800:
            # if so, refresh the token
            self.logger.info('TGT refresh required, requesting now..')
            return (True, self.requestTGT())
        else:
            self.logger.info('No TGT refresh needed')
            return (False, self.tgturl)

    def getSingleUseTicket(self):
        """ Requests and returns a single-use token with keys & creds stored in the class instance

        Returns:
            string: Single-use token
        """        

        self.ensureTGT()
        r = requests.post(self.tgturl, params={'service': 'http://umlsks.nlm.nih.gov'}, headers=UMLS_handler.HEADERS)

        if r.status_code == 200:
            self.logger.info('Successfully got Single-use ticket: HTTP ' + str(r.status_code))
            return r.text
        else:
            raise RuntimeError('Failed getting Single-use ticket: HTTP ' + str(r.status_code))

if __name__ == "__main__":
    extreme_gaming = UMLS_handler()
    extreme_ticket = extreme_gaming.getSingleUseTicket()
    print(extreme_ticket)