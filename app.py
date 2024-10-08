from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from pandas_geojson import read_geojson
import ydata_profiling
import os
import streamlit as st
from streamlit_option_menu import option_menu
from numpy import NaN
import base64
from components.data_editing import data_editing_page
from components.exploration import data_exploration_page
from components.feature_engineering import feature_engineering_page
from components.inference import inference_page
from components.logger import experiment_log_page, show_classification_metrics_logger, show_clustering_metrics_logger, show_regression_metrics_logger
from components.modeling import modeling_page, show_classification_metrics, show_clustering_metrics, show_data_prediction, show_regression_metrics
import util as utl
from util import config
from sqlalchemy import create_engine
from sqlalchemy.schema import Table, MetaData, Column
from sqlalchemy.sql.expression import select, text

from util import get_data, load_result, next_page, prev_page, save_result, show_roc_auc_score_binary_class

import pickle

import warnings
warnings.simplefilter(action='ignore')

# Page configuration
st.set_page_config(page_title="Data Modeler", layout="wide", page_icon='assets/kejaksaan2.png')

# Loading CSS
utl.local_css("assets/custom.css")

# Hiding hamburger menu from streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .appview-container > section[tabindex="0"] {
        background: bottom center/contain no-repeat url("data:image/png;base64,%s");
        background-color: #191a1dff;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('assets/bg.png')

# Logo in side bar configuration
st.sidebar.image("assets/kejaksaan2.png", output_format='PNG', width=200)

# Sidebar Menu
with st.sidebar:
    menu_selected = option_menu("", ["Home", "Data Exploration", "Feature Engineering", "Modelling", "Experiment Logs", "Inference"],
                                icons=["house", "card-list", "columns-gap", "gear", "folder", "play"],
                                menu_icon="cast",
                                default_index=0,
                                styles={
                                    "container": {"padding": "0!important", "background-color": "#25292aff"},
                                    "icon": {"color": "white", "font-size": "20px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#444444"},
                                    "nav-link-selected": {"color": "#FF7F00", "background-color": "rgba(128, 128, 128, 0.1)"}
    })

    # Adding Help Desk button
    st.markdown(
        f"""
        <style>
        .help-desk-button {{
            background-color: transparent;
            color: white;
            text-align: left;
            padding: 10px 15px;
            border: none;
            cursor: pointer;
            font-size: 15px;
            display: flex;
            align-items: center;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }}
        .help-desk-button:hover {{
            background-color: rgba(128, 128, 128, 0.1);
        }}
        .help-desk-icon {{
            font-size: 20px;
            margin-right: 15px;
            color: white;
        }}
        </style>
        <a href="{config['help_desk_url']}" target="_blank" style="text-decoration: none;">
            <div class="help-desk-button">
                <span class="help-desk-icon">&#x2753;</span> Help Desk
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

# Configuring home menu
if menu_selected == "Home":
    try:
        engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}")
        connection = engine.connect()
        connection.execute(f"CREATE DATABASE IF NOT EXISTS {config['db_name']};")
    except:
        st.warning("Database is not connected please check is your database is on, or reload the page")
    st.image("assets/header.png", output_format='PNG')

# Configuring data exploration menu
if menu_selected == "Data Exploration":
    data_exploration_page(st)

# Configuring Feature Engineering Menu
if menu_selected == "Feature Engineering":
    feature_engineering_page(st)

# Configuring Modelling Menu
if menu_selected == "Modelling":
    modeling_page(st)

# Configuring Experiment Logs Menu
if menu_selected == "Experiment Logs":
    experiment_log_page(st)

# Configuring Inference Menu
if menu_selected == "Inference":
    inference_page(st)
