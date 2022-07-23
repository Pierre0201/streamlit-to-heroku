# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:46:49 2022
@author: Pierre
"""
#import streamlit.components.v1 as components
#import time
#from matplotlib.patches import Rectangle

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from joblib import load
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

#path = 'C:/Users/Pierre/#P7 DS OC/'
path = 'https://github.com/Pierre0201/streamlit-to-heroku/'
#clf = load(path+'clf.joblib')
train_df = pd.read_csv(path+'train_df_dash.csv')
test_df = pd.read_csv(path+'submission_kernel02.csv')

feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

explainer = shap.TreeExplainer(clf, train_df[feats])
credit = 0


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


# Add a slider to the sidebar:
credit = st.sidebar.number_input("Enter credit application", value=int() ) 
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

delta = test_df['TARGET'].iloc[credit]-np.median(test_df['TARGET'])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Default Risk", "{:.2%}".format(test_df['TARGET'].iloc[credit]), "{:.2%}".format(delta), delta_color="inverse")
col2.metric("Minimum", "{:.2%}".format(min(test_df['TARGET'])))
col3.metric("Maximum", "{:.2%}".format(max(test_df['TARGET'])))
col4.metric("Median", "{:.2%}".format(np.median(test_df['TARGET'])))

liste = (f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'])
#liste = ['PAYMENT_RATE','DAYS_BIRTH','test1','test2']

#option = st.sidebar.selectbox('Select a first variable', (liste))
#st.sidebar.write('You selected:', option)

options = st.sidebar.multiselect('What variables do you choose', (liste), (['PAYMENT_RATE','DAYS_BIRTH']))

#st.sidebar.write('You selected:', options)

#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)

# Space out the maps so the first one is 2x the size of the other three
#c1, c2, c3, c4 = st.columns((2, 1, 1, 1))
#col1, col2 = st.columns([2,3])


#with col1:
#    st.header("Summary Plot")
#    shap.summary_plot(explainer.shap_values(test_df[feats]),
#                      features = test_df[feats],
#                      feature_names=feats)
#    st.pyplot(bbox_inches='tight')

#with col2:
#    st.header("Waterfall Plot")
#    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, 
#                                           explainer.shap_values(test_df[feats].iloc[credit]),
#                                           feature_names=feats
#                                          )
#    st.pyplot(bbox_inches='tight')


#col1, col2 = st.columns(2)

#with col1:   
#    st.header("Hist Plot")
#    fig = plt.figure(figsize=(9, 7))
#    bins = np.histogram_bin_edges(train_df[options[0]], bins='auto')
#    p = sns.histplot(data=train_df, x=options[0], hue="TARGET", stat="density", common_norm=False, bins=bins)
#    difference_array = np.absolute(bins-train_df[options[0]].iloc[credit])
#    for rectangle in p.patches:
#        if min(bins[difference_array.argsort()[:2]]) <= rectangle.get_x() < max(bins[difference_array.argsort()[:2]])  :
#            rectangle.set_facecolor('#ffd966')
#    st.pyplot(fig, bbox_inches='tight')

#with col2:
#    st.header("Bivariate Plot")
#    fig = plt.figure(figsize=(9, 7))
#    sns.scatterplot(x=options[0], y=options[1], data=train_df.sample(250), hue='TARGET')    
#    #plt.plot(0.3,0.5, marker="x", color="r")
#    st.pyplot(fig, bbox_inches='tight')
#    
#col1, col2 = st.columns([2,1])    
#vwith col1:
#    st.header("Bivariate Plot Bis")
#    sns.jointplot(data=train_df, x=options[0], y=options[1], kind='hex')
#    plt.plot(train_df[options[0]].iloc[credit],train_df[options[1]].iloc[credit], marker="H", color="r")
#    st.pyplot()
#
