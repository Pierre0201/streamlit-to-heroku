# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 00:40:59 2022

@author: Pierre
"""

import requests
import pandas as pd

HOST = 'https://fastapi-clf-predict.herokuapp.com/'



def get_prediction(credit_data: pd.Series):
    """
    Args:
        credit_data (pd.Series): credit data for one customer

    Returns: (float) probability of credit default, ranging from 0 to 1.
    """
    response = requests.post(HOST + 'prediction', data=credit_data.to_json())
    proba_default = response.json()["probability"]
    return proba_default

def minimum():
    response = requests.get(HOST + 'minimum/')
    minimum = eval(response.content)["min"]
    return minimum

def maximum():
    response = requests.get(HOST + 'maximum/')
    maximum = eval(response.content)["max"]
    return maximum

def median():
    response = requests.get(HOST + 'median/')
    median = eval(response.content)["median"]
    return median
