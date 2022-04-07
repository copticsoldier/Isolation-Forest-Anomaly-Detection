#This branch is for bringing in BiqQuery pulling capability to the streamlit anomalies app

import zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
import json
import streamlit as st
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq
from google.oauth2 import service_account


@st.cache(suppress_st_warning=True)
def load_csv():
    df_input=pd.DataFrame()
    df_input=pd.read_csv(input)
    return df_input
class IsolationForestModel:
    """
    This class runs the isolation forest algorithm on
    the specified variable name in the uploaded dataset
    """

    def __init__(self, key, n_estimators, verbose, max_samples, contamination, sensitivity):
        self.key = key
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.max_samples = max_samples
        self.contamination = contamination
        self.sensitivity = sensitivity
        self.data = df

    def _data_prep(self):
        try:
            self.data_subset = df[[str(column1),str(column2)]]
            if Key == None:
                for col in self.data_subset.columns:
                    if self.data_subset[col].dtype == object:
                        self.data_subset[col] = self.data_subset[col].astype('category')
                        self.data_subset[col] = self.data_subset[col].apply(lambda x: zlib.crc32(x.encode('utf-8')))
            else:
                self.data_subset = self.data_subset[self.data_subset[column1] == self.key]
                for col in self.data_subset.columns:
                    if self.data_subset[col].dtype == object:
                        self.data_subset[col] = self.data_subset[col].astype('category')
                        self.data_subset[col] = self.data_subset[col].apply(lambda x: zlib.crc32(x.encode('utf-8')))
                        
        except Exception as e:
            logging.error(traceback.print_tb(e.__traceback__))

    def _model_fit_predict(self):
        try:
            iso = IsolationForest(n_estimators=self.n_estimators,verbose=self.verbose,
                max_samples=self.max_samples, contamination=self.contamination)
            iso.fit(self.data_subset)
            self.decision_scores = iso.decision_function(self.data_subset)
            self.anom_scores = -1*iso.score_samples(self.data_subset)
            self.data_subset['Anomaly Scores'] = 1-self.anom_scores
            self.indices = self.data_subset.index[self.data_subset['Anomaly Scores']<= self.sensitivity]
        except Exception as e:
            logging.error(traceback.print_tb(e.__traceback__))

    def _anomaly_rows(self):
        self.anomaly_events = df.loc[self.indices]
        return st.write(self.anomaly_events)
        
    def _anom_scores_plot(self):
        fig, ax =plt.subplots(figsize=(20,12))
        ax.set_title('Distribution of Isolation Forest Scores', fontsize = 18, loc='center')
        ax.set_xlabel('Anomaly Scores',fontsize=16)
        ax.set_ylabel('Row Density',fontsize=16)
        ax = sns.distplot(self.decision_scores,kde_kws={"color": "blue", "lw": 2},hist_kws = {"alpha": 0.5,"color":"red"})
        return st.pyplot(fig)

    def run_model(self):
        IsolationForestModel._data_prep(self)
        IsolationForestModel._model_fit_predict(self)
        return IsolationForestModel._anomaly_rows(self), IsolationForestModel._anom_scores_plot(self)


st.title("Anomaly Detection Application")
st.write("Isolation Forest Model to Detect Anomalies in Data Points:\nSelect dataset to run anomalies model")
df =  pd.DataFrame()
st.subheader("1. Load the data")   
input = st.file_uploader('Drag and drop csv file here')

try:

    if input is None:
        st.write("Or use sample dataset to try the application")
        sample = st.checkbox("Download Phone Numbers Sample Data")
        bigquery = st.checkbox("Pull table from Google BigQuery")
        if sample:
            input = 'C:/Users/T460/Documents/Phone_numbers_sample.csv' 
            df = load_csv()
            st.write(df.head(10))
        if bigquery:
            filename = st.text_input('Enter path to Google service account Key:')
            sql = st.text_input('Create Query:')
            credentials = service_account.Credentials.from_service_account_file(filename)
            input = pandas_gbq.read_gbq(sql, credentials=credentials)
            df = input
            st.write(df.head(10))
    try:
        if sample:
            st.markdown("""[download_link](https://storage.googleapis.com/public-content-r6ns2jk5u99zr6mv7v4h-j4sp6x6tymwet9bfey0o/Phone_numbers_sample.csv)""")
    except:
        if input:
            with st.spinner('Loading data..'):
                df = load_csv()
                st.write(df.head(10))

    st.subheader("2. Choose the 2 columns that the model will use to check for anomalies")
    st.write("Select column that contains the variable name")
    column1=st.selectbox('Variable Name',df.columns)
    Keys = df[column1].unique()
    st.write("Select a value in the column which you'd like to subset by, otherwise press None")
    Key = st.selectbox('Key',np.append(Keys,None))
    st.write("Select column which may contain anomalous values")
    column2=st.selectbox('Variable Value',df.columns)
    st.subheader("3. Alter the sensitivity")
    st.write("Increasing the sensitivity will return more anomalous rows")
    Sensitivity = st.select_slider('Sensitivity',list(np.arange(0,1.0,0.01)))
    class_copy = IsolationForestModel(Key, 20, 2, 15000, 'auto',Sensitivity)
    st.write(class_copy.run_model())

except:
    None