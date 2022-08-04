# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 00:40:59 2022

@author: Pierre
"""

import requests

HOST = 'https://fastapi-clf-predict.herokuapp.com/'

def get_prediction(id_credit):
    response = requests.get(HOST + 'predict?id_credit=' + id_credit)
    proba_default = eval(response.content)["probability"]
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
