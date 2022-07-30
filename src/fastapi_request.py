# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 00:40:59 2022

@author: Pierre
"""
import requests
import pandas as pd

path = 'https://raw.githubusercontent.com/Pierre0201/streamlit-to-heroku/main/src/ressources/'
test_df = pd.read_csv(path+'submission_kernel02.csv')


HOST = 'https://fastapi-clf-predict.herokuapp.com/'

def get_prediction(id_credit):
    """Gets the probability of default of a client on the API server.
    Args : 
    - id_client (int).
    Returns :
    - probability of default (float).
    """
    json_client = test_df[feats].loc[test_df['SK_ID_CURR']==id_credit].to_json()
    response = requests.get(HOST + '/prediction/', data=json_client)
    proba_default = eval(response.content)["probability"]
    return proba_default
