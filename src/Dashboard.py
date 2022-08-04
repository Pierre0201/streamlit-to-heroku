# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:46:49 2022
@author: Pierre
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from joblib import load
import seaborn as sns
from io import BytesIO
import requests

from fastapi_request import get_prediction, minimum, maximum, median

st.set_option('deprecation.showPyplotGlobalUse', False)

path = 'https://raw.githubusercontent.com/Pierre0201/streamlit-to-heroku/main/src/ressources/'

mLink = 'https://github.com/Pierre0201/streamlit-to-heroku/blob/main/src/ressources/clf.joblib?raw=true'
mfile = BytesIO(requests.get(mLink).content)
clf = load(mfile)
    
train_df = pd.read_csv(path+'train_df_dash.csv')
test_df = pd.read_csv(path+'submission_kernel02.csv')

feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

explainer = shap.TreeExplainer(clf, train_df[feats])
credit = test_df.loc[test_df['SK_ID_CURR']==100141].index[0]-len(test_df)

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


# Add a slider to the sidebar:
id_credit = st.sidebar.number_input("Enter credit application", value=100141 ) 
credit =  test_df.loc[test_df['SK_ID_CURR']==id_credit].index[0] - len(test_df)

st.sidebar.write('Exemple:', list(test_df['SK_ID_CURR'].sample(3)))
shap_values = explainer.shap_values(test_df[feats].iloc[credit])


plt.style.use('fivethirtyeight')
plt.rcParams.update(
    {
     'xtick.labelsize':15,
     'ytick.labelsize':15,
     'axes.labelsize': 15,
     'legend.fontsize': 15,
     'axes.titlesize':15,
     'axes.titleweight':'bold',
     'axes.titleweight':'bold'
    })

seuil = 0.11
delta = get_prediction(str(id_credit))-seuil

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Default Risk", "{:.2%}".format(get_prediction(str(id_credit))), "{:.2%}".format(delta), delta_color="inverse")
col2.metric("Threshold","11%")
col3.metric("Minimum", "{:.2%}".format(minimum()))
col4.metric("Maximum", "{:.2%}".format(maximum()))
col5.metric("Median", "{:.2%}".format(median()))


if test_df['TARGET'].iloc[credit] > seuil:
    st.error('Decision : Refused')
if test_df['TARGET'].iloc[credit] <= seuil:
    st.success('Decision : Accepted')

liste = (f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'])
options = st.sidebar.multiselect('What variables do you choose', (liste), (['PAYMENT_RATE','DAYS_BIRTH']))

col1, col2 = st.columns(2)


with col1:
    st.header("Summary Plot")
    shap.summary_plot(explainer.shap_values(test_df[feats]),
                      features = test_df[feats],
                      feature_names=feats)
    st.pyplot(bbox_inches='tight')

with col2:
    st.header("Waterfall Plot")
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, 
                                           explainer.shap_values(test_df[feats].iloc[credit]),
                                           feature_names=feats
                                          )
    st.pyplot(bbox_inches='tight')


col1, col2 = st.columns(2)

with col1:   
    st.header("Hist Plot")
    fig = plt.figure(figsize=(9, 7))
    bins = np.histogram_bin_edges(train_df[options[0]], bins='auto')
    p = sns.histplot(data=train_df, x=options[0], hue="TARGET", stat="density", common_norm=False, bins=bins)
    difference_array = np.absolute(bins-train_df[options[0]].iloc[credit])
    for rectangle in p.patches:
        if min(bins[difference_array.argsort()[:2]]) <= rectangle.get_x() < max(bins[difference_array.argsort()[:2]])  :
            rectangle.set_facecolor('#ffd966')
    st.pyplot(fig, bbox_inches='tight')

with col2:
    st.header("Bivariate Plot")
    fig = plt.figure(figsize=(9, 7))
    sns.scatterplot(x=options[0], y=options[1], data=train_df.sample(250), hue='TARGET')    
    plt.plot(test_df[options[0]].iloc[credit],test_df[options[1]].iloc[credit], marker="x", color="r")
    st.pyplot(fig, bbox_inches='tight')
    
col1, col2 = st.columns(2)    
with col1:
    st.header("Bivariate Plot Bis")
    sns.jointplot(data=train_df, x=options[0], y=options[1], kind='hex')
    plt.plot(test_df[options[0]].iloc[credit],test_df[options[1]].iloc[credit], marker="H", color="r")
    st.pyplot()
with col2:
    st.header("Bivariate Plot Ter")
    sns.kdeplot(data=train_df, x=options[0], y=options[1], hue="TARGET", fill=True)
    plt.plot(test_df[options[0]].iloc[credit],test_df[options[1]].iloc[credit], marker="H", color="r")
    st.pyplot()
