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

def load_csv():
    df_input=pd.read_csv(input)
    return df_input

#First iteration of the app to make it applicable to a more datasets
class IsolationForestModel:
    """
    This class runs the isolation forest algorithm on
    the specified variable name in the adobe dataset
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
            anom_scores = iso.score_samples(self.data_subset)
            self.data_subset['Anomaly Scores'] = -1 * anom_scores
            self.indices = self.data_subset.index[self.data_subset['Anomaly Scores']>= self.sensitivity]
        except Exception as e:
            logging.error(traceback.print_tb(e.__traceback__))

    def _anomaly_rows(self):
        self.anomaly_events = df.loc[self.indices]
        return st.write(self.anomaly_events)
        
    def _anom_scores_plot(self):
        fig, ax =plt.subplots(figsize=(20,12))
        ax.set_title('Distribution of Isolation Forest Scores', fontsize = 15, loc='center')
        ax = sns.distplot(self.data_subset['Anomaly Scores'],color='red',hist_kws = {"alpha": 0.5})
        return st.pyplot(fig)

    def run_model(self):
        IsolationForestModel._data_prep(self)
        IsolationForestModel._model_fit_predict(self)
        return IsolationForestModel._anomaly_rows(self), IsolationForestModel._anom_scores_plot(self)


df=pd.DataFrame()
st.write("Isolation Forest Model to Detect Anomalies in Data Points:\nSelect dataset to run anomalies model")
input = st.file_uploader('Drag and drop csv file here')

if input is None:
    st.write("Or use sample dataset to try the application")
    sample = st.checkbox("Upload Adobe Data")
    if sample:
        input = 'C://Users//T460//Downloads//adobe-data.csv' 
        df = load_csv()
        st.write(df.head(10))

try:
    if sample:
        st.markdown("""[download_link]()""")
except:
    if input:
        with st.spinner('Loading data..'):
            df = load_csv()
            st.write(df.head(10))

st.write("Select column that contains the variable name")
column1=st.selectbox('Variable Name',df.columns)
st.write("Select column which may contain anomalous values")
column2=st.selectbox('Variable Value',df.columns)

Keys = df[column1].unique()
Key = st.selectbox('Key',np.append(Keys,None))
Sensitivity = st.select_slider('Sensitivity',list(np.arange(0,1.0,0.01)))
class_copy = IsolationForestModel(Key, 10, 2, 15000, 'auto',Sensitivity)
st.write(class_copy.run_model())