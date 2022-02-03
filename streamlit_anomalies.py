import zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
import json
import streamlit as st
import traceback

def load_csv():
    df_input=pd.read_csv(input)
    return df_input

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
            self.data_subset = df[['variable_name', 'variable_value']]
            # At the moment, this app will only cater to the adobe-data.csv dataset due to the defined
            # column titles above
            self.data_subset = self.data_subset[self.data_subset['variable_name'] == self.key]
            for col in self.data_subset.columns:
                self.data_subset[col] = self.data_subset[col].astype('category')
                self.data_subset[col] = self.data_subset[col].apply(
                    lambda x: zlib.crc32(x.encode('utf-8')))
        except Exception as e:
            logging.error(traceback.print_tb(e.__traceback__))

    def _model_fit_predict(self):
        try:
            iso = IsolationForest(n_estimators=self.n_estimators,verbose=self.verbose,
                max_samples=self.max_samples, contamination=self.contamination)
            iso.fit(self.data_subset)
            anom_scores = iso.score_samples(self.data_subset)
            self.data_subset['anomaly'] = -1 * anom_scores
            self.indices = self.data_subset.index[self.data_subset['anomaly']>= self.sensitivity]
        except Exception as e:
            logging.error(traceback.print_tb(e.__traceback__))

    def _anomaly_rows(self):
        self.anomaly_events = df.loc[self.indices]
        return st.write(self.anomaly_events)

    def run_model(self):
        IsolationForestModel._data_prep(self)
        IsolationForestModel._model_fit_predict(self)
        return IsolationForestModel._anomaly_rows(self)


st.write("Here's our first attempt at creating a streamlit app:\nSelect Dataset to run anomalies model")

input = st.file_uploader('Drag and drop csv file here')

if input is None:
    st.write("Or use sample dataset to try the application")
    sample = st.checkbox("Upload Adobe Data")
    input = 'C://Users//T460//Downloads//adobe-data.csv' 
    df = load_csv()
try:
    if sample:
        st.markdown("""[download_link]()""")
except:
    if input:
        with st.spinner('Loading data..'):
            df = load_csv()
            st.write("Columns:")
            st.write(list(df.columns))
            columns = list(df.columns)

Key = st.selectbox('Key',df['variable_name'].unique())
Sensitivity = st.select_slider('Sensitivity',list(np.arange(0,1.0,0.01)))
class_copy = IsolationForestModel(Key, 10, 2, 15000, 'auto',Sensitivity)
rows = class_copy.run_model()
st.write(rows)




