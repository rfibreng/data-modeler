import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from streamlit_option_menu import option_menu
from util import load_result, prev_page, next_page
import os
import pickle
from datetime import datetime, timedelta
from util import config

from components.modeling import show_classification_report, show_confussion_matrix, show_data_prediction, show_roc_auc_report

def show_classification_metrics_logger(data_train_full_prediction, data_test_full_prediction,
                           classification_report_train, classification_report_test,
                           roc_auc_train, roc_auc_test, cm, label_target, model_selection, config, uid):
    st.markdown(f"<h3 class='menu-secondary'>Model: {model_selection}</h3>",
                                unsafe_allow_html=True)
    st.markdown(f"<h4>Model Id: {uid}</h4>",
                                unsafe_allow_html=True)
    st.markdown(f"<h4>Model Configuration</h4>",
                                unsafe_allow_html=True)
    for key, value in config.items():
        st.write(f'{key}: {value}')
    show_data_prediction(data_train_full_prediction, data_test_full_prediction)

    # Giving space
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing score
    show_classification_report(classification_report_train, classification_report_test)

    # Giving two spaces
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing ROC-AUC Score
    show_roc_auc_report(roc_auc_train, roc_auc_test)

    # Giving two spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing Confusion Matrix
    show_confussion_matrix(cm, label_target)

def show_regression_metrics_logger(data_train_full_prediction, data_test_full_prediction, mae_train, 
                                   mae_test, mse_train, mse_test, model_selection, config, uid):
    st.markdown(f"<h3 class='menu-secondary'>Model: {model_selection}</h3>",
                                unsafe_allow_html=True)
    st.markdown(f"<h4>Model Id: {uid}</h4>",
                                unsafe_allow_html=True)
    st.markdown(f"<h4>Model Configuration</h4>",
                                unsafe_allow_html=True)
    for key, value in config.items():
        st.write(f'{key}: {value}')
    show_data_prediction(data_train_full_prediction, data_test_full_prediction)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing mean absolute score
    st.write("Train Score")
    st.write(mae_train)
    st.write("Test Score")
    st.write(mae_test)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing score of MSE
    st.write("Train Score")
    st.write(mse_train)
    st.write("Test Score")
    st.write(mse_test)

    # Giving two spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

def show_clustering_metrics_logger(data_full_clustered, calinski_harabasz, davies_bouldin, model_selection, config,uid):
    st.markdown(f"<h3 class='menu-secondary'>Model: {model_selection}</h3>",
                                unsafe_allow_html=True)
    st.markdown(f"<h4>Model Id: {uid}</h4>",
                                unsafe_allow_html=True)
    st.markdown(f"<h4>Model Configuration</h4>",
                                unsafe_allow_html=True)
    for key, value in config.items():
        st.write(f'{key}: {value}')
    show_data_prediction(data_full_clustered)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    st.write("Calinski-Harabasz Index")
    st.write(calinski_harabasz)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    st.write("Davies-Bouldin Index")
    st.write(davies_bouldin)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

def filter_files(filename, start_date, end_date, model_name):
    data = filename.split('_')
    if len(data) < 3:
        return False
    uid, timestamp, model_selection = data[0], data[1], data[2]
    timestamp = timestamp.split('.')[0]
    file_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  # Adjust format as necessary
    return (start_date <= file_date.date() <= end_date) and (model_name.lower() in model_selection.lower())

def experiment_log_page(st):
    st.markdown("<h2 class='menu-title'>Experiment Logs</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Access your experiments logs here</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)
    task_selected = option_menu("", ["Classification", "Regression", "Clustering"],
                                icons=["house", "card-list", "award"],
                                menu_icon="cast",
                                orientation="horizontal",
                                default_index=0,
                                styles={
                                    "container": {"background-color": "#25292aff"},
                                    # "icon": {"color": "orange", "font-size": "25px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#444444", "text-align-last": "center"},
                                    "nav-link-selected": {"color": "#FF7F00", "background-color": "rgba(128, 128, 128, 0.1)"}
    })

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)
    # Input for start date, end date, and model name filtering
    col1,col2,col3 = st.columns(3)

    hundred_years_ago = datetime.now() - timedelta(days=365 * 100)
    start_date = col1.date_input("Start date", value=hundred_years_ago)
    end_date = col2.date_input("End date")
    model_name = col3.text_input("Filter by model name")

    st.markdown("<br>", unsafe_allow_html=True)
    print(config["output_path"])
    result_root_path = f'{config["output_path"]}/default'
    page_limit = 25
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
        st.session_state.current_task=task_selected
    else:
        if st.session_state.current_task != task_selected:
            st.session_state.current_page = 0
            st.session_state.current_task=task_selected
            
    if os.path.exists(result_root_path):
        list_directory = os.listdir(os.path.join(result_root_path, task_selected))
        list_directory.sort(key=lambda x: x.split('_')[1], reverse=True)
        list_directory = [f for f in list_directory if filter_files(f, start_date, end_date, model_name)]
        col1,col2,col3 = st.columns([1,3,1])

        start_index = st.session_state.current_page * page_limit
        end_index = start_index + page_limit
        list_directory_filtered = list_directory[start_index:end_index]

        if st.session_state.current_page > 0:
            col1.button('Previous', on_click=lambda: prev_page(st))
        if end_index < len(list_directory):
            col3.button('Next', on_click=lambda: next_page(st))
        else:
            st.markdown('<br>', unsafe_allow_html=True)
        print(st.session_state.current_page)

        col1,col2,col3,col4 = st.columns([1,3,3,3])
        col1.markdown("#### No.")
        col2.markdown("#### Datetime")
        col3.markdown("#### Model")
        col4.markdown("#### Download")
        for count,ld in enumerate(list_directory_filtered):
            data = ld.split('_')
            
            if len(data)<3:
                continue
            
            folder = os.path.join(result_root_path, task_selected, ld)
            uid, timestamp, model_selection = data
            col1,col2,col3,col4 = st.columns([1,3,3,3])
            col1.text(count+1+start_index)
            col2.text(timestamp)
            col3.text(model_selection)
            col4.download_button(label="Download", key=uid, data=pickle.dumps(load_result(os.path.join(folder, "model.pickle"))), file_name=f'{ld}.pickle')
            if task_selected=='Classification':
                with st.expander("Show Data"):
                    data_to_load = load_result(os.path.join(folder, "output_prediction.pickle"))
                    label_target = data_to_load['outputs']['y_train_predict']['Target Actual'].unique()
                    show_classification_metrics_logger(data_to_load['outputs']['y_train_predict'], data_to_load['outputs']['y_test_predict'], 
                                                data_to_load['outputs']['classification_score_train'], data_to_load['outputs']['classification_score_test'], 
                                                data_to_load['outputs']['roc_auc_train'], data_to_load['outputs']['roc_auc_test'],
                                                data_to_load['outputs']['confussion_matrix'], label_target, model_selection, data_to_load['configuration'], uid)
            elif task_selected=='Regression':
                with st.expander("Show Data"):
                    data_to_load = load_result(os.path.join(folder, "output_prediction.pickle"))
                    show_regression_metrics_logger(data_to_load['outputs']['y_train_predict'], data_to_load['outputs']['y_test_predict'],
                                                data_to_load['outputs']['mae_train'], data_to_load['outputs']['mae_test'],
                                                data_to_load['outputs']['mse_train'], data_to_load['outputs']['mse_test'],
                                                model_selection, data_to_load['configuration'], uid)
            elif task_selected=='Clustering':
                with st.expander("Show Data"):
                    data_to_load = load_result(os.path.join(folder, "output_prediction.pickle"))
                    show_clustering_metrics_logger(data_to_load['outputs']['data_full_clustered'], data_to_load['outputs']['calinski_harabasz'],
                                                data_to_load['outputs']['davies_bouldin'], model_selection, data_to_load['configuration'], uid)
    else:
        col1,col2,col3,col4 = st.columns([1,3,3,3])
        col1.markdown("#### No.")
        col2.markdown("#### Datetime")
        col3.markdown("#### Model")
        col4.markdown("#### Download")