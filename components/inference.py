from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from util import get_data, load_result
from streamlit_option_menu import option_menu
import os
import pickle
from util import config, remove_punctuation
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_values
from datetime import datetime
import uuid
from psycopg2.extensions import AsIs
import pandas as pd

def dtype_to_sql(dtype):
    if dtype == 'int64':
        return 'BIGINT'
    elif dtype == 'float64':
        return 'DOUBLE PRECISION'
    elif dtype == 'bool':
        return 'BOOLEAN'
    else:
        return 'TEXT'

def insert_data(df, dataset_name, model_description, conn):
    # Generate a unique identifier
    uid = uuid.uuid4()
    uid = str(uid).replace("-","")

    # Establish a connection to the database
    cursor = conn.cursor()

    # Construct table name
    table_name = f"predictions_{uid}_{dataset_name.replace('.csv','')}_output_prediction"
    
    # Create a new table based on the structure of df
    cols_with_types = ", ".join([f"{remove_punctuation(col).replace(' ','_')} {dtype_to_sql(df[col].dtype.name)}" for col in df.columns if not col.startswith('Unnamed')])
    create_table_query = f"CREATE TABLE {table_name} ({cols_with_types})"
    cursor.execute(create_table_query)

    # Prepare and execute the insertion query for the main predictions_master table
    cursor.execute(
        "INSERT INTO predictions_master (id, dataset, model_prediction, prediction_at) VALUES (%s, %s, %s, %s)",
        (str(uid), dataset_name, model_description, datetime.now())
    )

    # Insert DataFrame data into the newly created table using psycopg2's execute_values for bulk insertion
    insert_query = f"INSERT INTO {table_name} ({', '.join([remove_punctuation(col).replace(' ','_') for col in df.columns if not col.startswith('Unnamed')])}) VALUES %s"
    execute_values(cursor, insert_query, df[[col for col in df.columns if not col.startswith('Unnamed')]].values.tolist())

    # Commit the transactions
    conn.commit()
    
    # Close the cursor and connection
    cursor.close()
    conn.close()
    
    print("Data input successfully")

def inference_page(st):

    conn = psycopg2.connect(dbname='postgres', user=config['db_user'], password=config['db_password'], host=config['db_host'])
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.close()
    conn.close()

    conn = psycopg2.connect(dbname='postgres', user=config['db_user'], password=config['db_password'], host=config['db_host'])
    cursor = conn.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions_master (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        dataset VARCHAR(50),
        model_prediction VARCHAR(100),
        prediction_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    conn.commit()  # Commit the transaction
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

                insert_data(dataframe, uploaded_file.name, model_name, conn)  # Specify your dataset name and model description

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h4 class='menu-secondary'>Original Data with Prediction</h3>",
                            unsafe_allow_html=True)
                st.write(dataframe)
                st.write("- The shape of data", dataframe.shape)
            except:
                st.warning("The model is not for that data, please make sure that the data is fitted with the model")