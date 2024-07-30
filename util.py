import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import os
import pickle
import yaml
import string
import random
import streamlit as st

def remove_punctuation(text):
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method to remove punctuation
    return text.translate(translator)

# Caching data for dataframe
@st.cache_data
def get_data(X, delimiter=','):
    df = pd.read_csv(X, encoding='ISO-8859-1', on_bad_lines='skip', delimiter=delimiter)
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
    uid = str(generate_random_alphanumeric())
    try:
        # Building the path using the config
        base_output_path = config['output_path']

        # Ensure the paths are combined correctly and handle different cases
        model_name = f'{uid}_{data_to_save["experiment_date"]}_{data_to_save["model_selection"]}'
        folder = os.path.join(base_output_path, 'default', model_selection, model_name)
        os.makedirs(folder, exist_ok=True)

        # Saving the prediction data
        with open(os.path.join(folder, 'output_prediction.pickle'), 'wb') as f:
            pickle.dump(data_to_save, f)

        # Saving the model
        with open(os.path.join(folder, 'model.pickle'), 'wb') as f:
            pickle.dump(model, f)
        
        st.success(f"Model saved into repository with model name: {model_name}")

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

def process_data(uploaded_file, option_selected, delimiter):
    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file, delimiter)

            # Initiating data on session state
            if "data" not in st.session_state:
                st.session_state['data'] = dataframe
                st.session_state['uploaded_file'] = dataframe
                if option_selected == "Upload":
                    st.session_state['data_name'] = uploaded_file.name
                else:
                    st.session_state['data_name'] = uploaded_file.split('/')[-1]
            else:
                if st.button('Update Data'):
                    st.session_state['data'] = dataframe
                    st.session_state['uploaded_file'] = dataframe   
                    if option_selected == "Upload":
                        st.session_state['data_name'] = uploaded_file.name
                    else:
                        st.session_state['data_name'] = uploaded_file.split('/')[-1]

        except Exception as e:
            print("waduh: ",e)
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

def data_uploader_components(st):
    option_selected = st.selectbox("Data Source", ['Upload','Data Studio'])
    delimiter = st.selectbox('delimiter', [',',';'])
    uploaded_file = None
    if option_selected == "Upload":
        uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                     type="csv",
                                     help="The supported file is only in csv formatted"
                                     )
        
    elif option_selected == "Data Studio":
        data_list = ["None"]
        # Use os.walk to traverse through all directories and subdirectories
        for dirpath, dirnames, filenames in os.walk(config['pds_data_path']):
            # Filter and add CSV files to data_list
            data_list.extend([os.path.join(dirpath, file) for file in filenames if file.endswith('.csv')])
        uploaded_file = st.selectbox("Data",data_list)
    process_data(uploaded_file, option_selected, delimiter)
    
def dtype_to_sql(dtype):
    if dtype == 'int64':
        return 'BIGINT'
    elif dtype == 'float64':
        return 'TEXT'
    elif dtype == 'bool':
        return 'BOOLEAN'
    else:
        return 'TEXT'
    
def add_index_column_if_first_is_float(df):
    """
    Checks if the first column of the DataFrame is a float type.
    If it is, adds a new column at the start of the DataFrame with an integer index starting at 1.

    Args:
    df (pd.DataFrame): The DataFrame to check and modify.

    Returns:
    pd.DataFrame: The modified DataFrame with a new index column added if the first column is float.
    """
    # Check if the first column is a float
    first_column = df.columns[0]
    if df[first_column].dtype == 'float' or df[first_column].dtype == 'float64':
        # Add a new column with an integer index starting from 1
        df.insert(0, 'index_column', range(1, len(df) + 1))
    return df
    
def transform_digits(text):
    if text.isdigit():
        text = '_'+text
    
    return text

def descriptive_stats_to_table(df):
    """
    Converts descriptive statistics of a DataFrame into a tabular format where each row corresponds
    to a statistical measure (like mean, std, etc.) for all variables, including a 'desc' column.

    Parameters:
        df (pd.DataFrame): The DataFrame for which to compute the descriptive statistics.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a different statistical measure,
                      with an additional 'desc' column.
    """
    # Computing the descriptive statistics
    descriptive_stats = df.describe().transpose()

    # Resetting index to add 'desc' column
    descriptive_stats.reset_index(inplace=True)
    descriptive_stats.rename(columns={'index': 'descr'}, inplace=True)

    return descriptive_stats

def correlation_matrix_to_table(df):
    """
    Converts a DataFrame's correlation matrix into a tabular format where each row corresponds
    to the correlations of one variable with all others, including a 'correlated by' column.

    Parameters:
        df (pd.DataFrame): The DataFrame for which to compute the correlation.

    Returns:
        pd.DataFrame: A DataFrame where each row shows correlations of one variable with others,
                      with an additional 'correlated by' column.
    """
    # Computing the correlation matrix
    correlation_matrix = df.corr()

    # Resetting index to add 'correlated by' column
    correlation_matrix.reset_index(inplace=True)
    correlation_matrix.rename(columns={'index': 'correlated by'}, inplace=True)

    return correlation_matrix
    
def generate_random_alphanumeric():
    characters = string.ascii_letters + string.digits  # Includes both letters and digits
    random_string = ''.join(random.choice(characters) for _ in range(5))
    return random_string