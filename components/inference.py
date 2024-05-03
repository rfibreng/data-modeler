from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from util import add_index_column_if_first_is_float, dtype_to_sql, get_data, load_result, transform_digits
from streamlit_option_menu import option_menu
import os
import pickle
from util import config, remove_punctuation
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime
from datetime import datetime
import uuid
import pandas as pd

def insert_data(df, dataset_name, model_description):
    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()

    # Generate a unique identifier
    uid = uuid.uuid4()
    uid = str(uid).replace("-","")

    # Construct table name
    table_name = f"predictions_{uid}_{dataset_name.replace('.csv','')}"

    valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
    df = df[valid_columns]
    df = add_index_column_if_first_is_float(df)
    
    # Create a new table based on the structure of df
    cols_with_types = ", ".join([f"{transform_digits(remove_punctuation(col).replace(' ','_'))} {dtype_to_sql(df[col].dtype.name)}" for col in df.columns if not col.startswith('Unnamed')])
    create_table_query = f"CREATE TABLE {table_name} ({cols_with_types})"
    connection.execute(create_table_query)

    # Prepare and execute the insertion query for the main predictions_master table
    connection.execute(
        "INSERT INTO predictions_master (id, dataset, model_prediction, prediction_at) VALUES (%s, %s, %s, %s)",
        (str(uid), dataset_name, model_description, datetime.now())
    )

    # Insert DataFrame data into the newly created table using psycopg2's execute_values for bulk insertion
    column_names = ', '.join([transform_digits(remove_punctuation(col).replace(' ','_')) for col in df.columns if not col.startswith('Unnamed')])
    placeholders = ', '.join(['%s'] * len(valid_columns))
    insert_query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
    values_to_insert = df[[col for col in df.columns if not col.startswith('Unnamed')]].values.tolist()
    connection.execute(insert_query, values_to_insert)
    
    print("Data input successfully")

def inference_page(st):

    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()

    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions_master (
        id VARCHAR(100),
        dataset VARCHAR(50),
        model_prediction VARCHAR(100),
        prediction_at DATETIME
    );
    """
    connection.execute(create_table_query)

    print("Database and table checked/created successfully.")

    st.markdown("<h2 class='menu-title'>Load Model</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Retrieving a trained machine learning model from storage and making it available to build predictions</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)
    
    option_selected = option_menu("", ["Experiment Logs", "Upload"],
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
    model_name = ''
    if option_selected == "Experiment Logs":
        result_root_path = f"{config['output_path']}/default"
        task_selected = st.selectbox("Task",["Classification", "Regression", "Clustering"])
        if os.path.exists(os.path.join(result_root_path, task_selected)):
            list_directory = os.listdir(os.path.join(result_root_path, task_selected))
            list_directory.sort(key=lambda x: x.split('_')[1], reverse=True)

            model_selected = st.selectbox("Model", list_directory)
            model = load_result(os.path.join(result_root_path, task_selected, model_selected, 'model.pickle'))
            model_name = model_selected
        else:
            st.warning("there's no model in repository")

    if option_selected == "Upload":
        st.markdown("<h4 class='menu-secondary'>Upload Model</h4>",
                    unsafe_allow_html=True)

        # Upload variable for uploading model
        uploaded_model = st.file_uploader("Choose a model to upload for making predictions",
                                        type=["pkl","pickle"],
                                        help="The supported file is only in pkl and pickle formatted",
                                        )
        # Setting the upload options when there's file on uploader menu
        if uploaded_model is not None:
            model_name = uploaded_model.name
            model = pickle.loads(uploaded_model.read())
            st.success("Model Loaded")

        else:
            st.write("")

    # Adding three spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 class='menu-secondary'>Upload File</h3>",
                unsafe_allow_html=True)

    # Upload variable for uploading file to be predicted
    uploaded_file = st.file_uploader("Choose a file to be predicted with machine learning model",
                                     type="csv",
                                     help="The supported file is only in csv formatted",
                                     )

    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file)
            st.success("The data have been successfully uploaded")

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 class='menu-secondary'>Original Data</h3>",
                        unsafe_allow_html=True)
            st.write(dataframe)
            st.write("- The shape of data", dataframe.shape)

        except:
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Make Prediction"):
            try:
                if isinstance(model, Pipeline):
                    if isinstance(model[1], DBSCAN) or isinstance(model[1],AgglomerativeClustering):
                        clusters = model.fit_predict(dataframe)
                    elif isinstance(model[1], Pipeline):
                        if isinstance(model[1][1], DBSCAN) or isinstance(model[1][1],AgglomerativeClustering):
                            clusters = model.fit_predict(dataframe)
                        else:
                            clusters = model.predict(dataframe)
                    else:
                        clusters = model.predict(dataframe)
                else:
                    clusters = model.predict(dataframe)

                # Adding cluster into data
                dataframe['Prediction'] = clusters

                model_name = task_selected + '-' + model_name

                insert_data(dataframe, uploaded_file.name, model_name)  # Specify your dataset name and model description

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h4 class='menu-secondary'>Original Data with Prediction</h3>",
                            unsafe_allow_html=True)
                st.write(dataframe)
                st.write("- The shape of data", dataframe.shape)
            except:
                st.warning("The model is not for that data, please make sure that the data is fitted with the model")