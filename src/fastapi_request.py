# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 00:40:59 2022

@author: Pierre
"""


import requests

MODEL_API_URL = 'https://fastapi-clf-predict.herokuapp.com/predict2'


def predict():
    try:
        response = requests.post(
            url=MODEL_API_URL,
            headers={'content-type': 'application/json'}           
        )
        response.raise_for_status()
        output = response.json()['prediction']
    except (requests.HTTPError, IOError) as err:
        output = str(err)
    return output

if __name__ == '__main__':
    # Example of model prediction
    print(predict())