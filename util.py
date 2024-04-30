import uuid
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import os
import pickle
from uuid import uuid4
import yaml
import string

def remove_punctuation(text):
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method to remove punctuation
    return text.translate(translator)

# Caching data for dataframe
@st.cache_data
def get_data(X):
    df = pd.read_csv(X)
    return df


@st.cache_data
def callback():
    edited_rows = st.session_state["data_editor"]["edited_rows"]
    rows_to_delete = []

    for idx, value in edited_rows.items():
        if value["x"] is True:
            rows_to_delete.append(idx)

    st.session_state["data"] = (
        st.session_state["data"].drop(
            rows_to_delete, axis=0).reset_index(drop=True)
    )


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def show_roc_auc_score_binary_class(y, y_predict_proba):
    return roc_auc_score(y, y_predict_proba[1])


def show_roc_auc_score_multi_class(y, y_predict_proba):
    return roc_auc_score(y, y_predict_proba, multi_class='ovr')


# Function to plot confusion matrix
def plot_confusion_matrix_multi_class(y_test, y_test_predict, label_target):
    st.markdown("<h4 class='menu-secondary'>Confusion Matrix Score</h4>",
                unsafe_allow_html=True)

    cm = confusion_matrix(
        y_test, y_test_predict, labels=label_target)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_target)
    disp.plot()
    st.pyplot(plt.show())

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config/config.yml')

def save_result(data_to_save, model_selection, model):
    uid = str(uuid4())
    try:
        # Building the path using the config
        base_output_path = config['output_path']

        # Ensure the paths are combined correctly and handle different cases

        folder = os.path.join(base_output_path, 'default', model_selection, f'{uid}_{data_to_save["experiment_date"]}_{data_to_save["model_selection"]}')
        os.makedirs(folder, exist_ok=True)

        # Saving the prediction data
        with open(os.path.join(folder, 'output_prediction.pickle'), 'wb') as f:
            pickle.dump(data_to_save, f)

        # Saving the model
        with open(os.path.join(folder, 'model.pickle'), 'wb') as f:
            pickle.dump(model, f)

    except Exception as e:
        print(e)

def load_result(path):
    return pickle.load(open(path, 'rb'))

# @st.cache_data
def next_page(st):
    st.session_state.current_page+=1
# @st.cache_data
def prev_page(st):
    st.session_state.current_page-=1

def next_page_inference(st):
    st.session_state.current_page_inference+=1
# @st.cache_data
def prev_page_inference(st):
    st.session_state.current_page_inference-=1

def process_data(uploaded_file, option_selected):
    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file)

            # Initiating data on session state
            if "data" not in st.session_state:
                st.session_state['data'] = dataframe
                st.session_state['uploaded_file'] = dataframe
                if option_selected == "Upload":
                    st.session_state['data_name'] = uploaded_file.name
                else:
                    st.session_state['data_name'] = uploaded_file
            else:
                if st.button('Update Data'):
                    st.session_state['data'] = dataframe
                    st.session_state['uploaded_file'] = dataframe   
                    if option_selected == "Upload":
                        st.session_state['data_name'] = uploaded_file.name
                    else:
                        st.session_state['data_name'] = uploaded_file

        except Exception as e:
            print("waduh: ",e)
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

def data_uploader_components(st):
    option_selected = st.selectbox("Data Source", ['Upload','PDS'])

    uploaded_file = None
    if option_selected == "Upload":
        uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                     type="csv",
                                     help="The supported file is only in csv formatted"
                                     )
        
    elif option_selected == "PDS":
        data_list = ["None"]+[x for x in os.listdir(config['pds_data_path']) if x.endswith('csv')]
        uploaded_file = st.selectbox("Data",data_list)
        uploaded_file = os.path.join(config['pds_data_path'], uploaded_file)
    process_data(uploaded_file, option_selected)
    
    