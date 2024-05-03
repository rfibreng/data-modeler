from util import add_index_column_if_first_is_float, data_uploader_components, dtype_to_sql, generate_random_alphanumeric, get_data, transform_digits
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine
import json
from util import config, remove_punctuation
from datetime import datetime
import uuid
import pandas as pd
import streamlit as st

def insert_data(dataset_name, df_output_features, is_scale, df_is_null, null_method, df_column_number_feature, df_column_text_feature, df_target_column):
    # Convert DataFrames to JSON strings as necessary
    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()

    uid = generate_random_alphanumeric()
    uid = str(uid).replace("-","")


    output_feature = {}
    for key,value in df_output_features.items():
        if value.isna().any().any():  # Check if there are any NaN values
            value = value.fillna('')
        output_feature[key] = value


    table_name = {key:f"fe_{uid}_{key}_{dataset_name.replace('.csv','')}" for key in output_feature.keys()}

    is_null = df_is_null if df_is_null is not None else None
    if is_null is not None:
        table_name['isnull'] = f"fe_{uid}_isnull_{dataset_name.replace('.csv','')}"
        output_feature['isnull'] = is_null.to_frame().T

    for key in table_name.keys():
        df = output_feature[key]
        if isinstance(df, pd.Series):
            clean_name = transform_digits(remove_punctuation(df.name).replace(' ','_'))
            sql_type = dtype_to_sql(df.dtype.name)
            col_with_type = f"{clean_name} {sql_type}"
            create_table_query = f"CREATE TABLE {table_name[key]} ({col_with_type})"
            print(col_with_type)
        else:
            df.columns = [transform_digits(remove_punctuation(str(col)).replace(' ', '_')) for col in df.columns]
            valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
            df = df[valid_columns]
            df = add_index_column_if_first_is_float(df)
            output_feature[key] = df
            cols_with_types = ", ".join([f"{transform_digits(remove_punctuation(col).replace(' ','_'))} {dtype_to_sql(df[col].dtype.name)}" for col in df.columns if not col.startswith('Unnamed')])
            create_table_query = f"CREATE TABLE {table_name[key]} ({cols_with_types})"
        connection.execute(create_table_query)

    column_number_feature = json.dumps(df_column_number_feature) if df_column_number_feature is not None else None
    column_text_feature = json.dumps(df_column_text_feature) if df_column_text_feature is not None else None
    target_column = json.dumps([df_target_column]) if df_target_column is not None else None

    # Prepare and execute the insertion query
    connection.execute(
        """
        INSERT INTO feature_engineering_master (
            id,dataset, is_scale, null_method,
            column_number_feature, column_text_feature, target_column, prediction_at
        ) VALUES (%s,%s, %s, %s, %s, %s, %s, %s)
        """,
        (uid,dataset_name, is_scale, null_method,
         column_number_feature, column_text_feature, target_column, datetime.now())
    )

    for key in table_name.keys():
        df = output_feature[key]
        if isinstance(df, pd.Series):
            column_name = transform_digits(remove_punctuation(df.name).replace(' ','_'))
            insert_query = f"INSERT INTO {table_name[key]} ({column_name}) VALUES (%s)"
            values_to_insert = [(value,) for value in df]
        else:
            df.columns = [transform_digits(remove_punctuation(str(col)).replace(' ', '_')) for col in df.columns]
            valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
            column_names = ', '.join([transform_digits(remove_punctuation(col).replace(' ','_')) for col in df.columns if not col.startswith('Unnamed')])
            placeholders = ', '.join(['%s'] * len(valid_columns))
            insert_query = f"INSERT INTO {table_name[key]} ({column_names}) VALUES ({placeholders})"
            values_to_insert = df[[col for col in df.columns if not col.startswith('Unnamed')]].values.tolist()
        connection.execute(insert_query, values_to_insert)
    
    print("Data input successfully")
    st.success(f"Data saved into database with table name:\n{'\n'.join([table_name[key] for key in table_name.keys()])}")

def feature_engineering_page(st):
    st.markdown("<h2 class='menu-title'>Feature Engineering</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Transforming raw data into a structured and usable format for training machine learning</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    if 'feature_data' not in st.session_state:
        st.session_state['feature_data'] = ""

    if 'target_data' not in st.session_state:
        st.session_state['target_data'] = ""

    if 'feature_data_train' not in st.session_state:
        st.session_state['feature_data_train'] = ""

    if 'feature_data_test' not in st.session_state:
        st.session_state['feature_data_test'] = ""

    if 'scaled_data_train' not in st.session_state:
        st.session_state['scaled_data_train'] = ""

    if 'scaled_data_test' not in st.session_state:
        st.session_state['scaled_data_test'] = ""

    if 'y_train' not in st.session_state:
        st.session_state['y_train'] = ""

    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = ""

    if 'classification_type' not in st.session_state:
        st.session_state['classification_type'] = ""

    # Making task option menu for feature engineering
    task_selected = option_menu("", ["Feature Engineering for Classification",
                                     "Feature Engineering for Regression",
                                     "Feature Engineering for Clustering"],
                                icons=["motherboard", "people"],
                                menu_icon="cast",
                                orientation="horizontal",
                                default_index=0,
                                styles={
                                    "container": {"background-color": "#25292aff"},
                                    # "icon": {"color": "orange", "font-size": "25px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#444444", "text-align-last": "center"},
                                    "nav-link-selected": {"color": "#FF7F00", "background-color": "rgba(128, 128, 128, 0.1)"}
    })

    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()

    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS feature_engineering_master (
        id VARCHAR(100),
        dataset VARCHAR(50),
        is_scale BOOLEAN,
        null_method VARCHAR(50) NULL,
        column_number_feature JSON NULL,
        column_text_feature JSON NULL,
        target_column JSON NULL,
        prediction_at DATETIME
    );
    """
    connection.execute(create_table_query)
    print("Database and table checked/created successfully.")

    # Setting engineering for Classification
    if task_selected == "Feature Engineering for Classification":
        is_scale = False
        is_null = None
        null_method = None
        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Assigning upload file variable
        data_uploader_components(st)

        # Menu if data already stored in session state for classification/regression
        if 'uploaded_file' in st.session_state:
            st.session_state['data'] = st.session_state['uploaded_file']
            pilihan_kolom = list(st.session_state.data.columns)

            # Making column for selecting feature and target
            col1, col2 = st.columns(2)

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            # Assigning option for feature column
            with col1:
                st.markdown("<br>", unsafe_allow_html=True)
                feature_column_number = st.multiselect("Select number column as feature",
                                                       st.session_state.data.columns,
                                                       default=list(
                                                           st.session_state.data.columns),
                                                       placeholder="Select columns")

             # Assigning option for target column
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                target_column = st.selectbox("Select column to be the target",
                                             st.session_state.data.columns)

            col3, col4 = st.columns(2)

            # Assigning option for target column
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                feature_column_text = st.multiselect("Select text column as feature",
                                                     st.session_state.data.columns,
                                                     default=None)

            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                classification_type = st.selectbox("Select classification type",
                                                   ["Binary Class", "Multi Class"])

            # Making column for showing features and target
            col5, col6 = st.columns([3, 1])

            full_data_feature = pd.concat([st.session_state.data[feature_column_number],
                                           st.session_state.data[feature_column_text]],
                                          axis=1)

            with col5:
                st.write("List of Feature Data")
                st.write(full_data_feature)
                st.write("- The shape of feature column : ",
                         full_data_feature.shape)

            with col6:
                st.write("Target Data")
                st.write(st.session_state.data[target_column])

            st.session_state['feature_data'] = full_data_feature
            st.session_state['target_data'] = st.session_state.data[target_column]
            # Checking for null values
            if st.checkbox("Check is_null data"):
                # Show if there is null data in the dataset
                null_data = st.session_state.data.isnull().sum()
                is_null = null_data
                st.write("Null values in each column:")
                st.write(null_data)

                # Handling null data options
                options = st.selectbox("Select how to handle null data:",
                                    ["Not handling", "Remove", "Mean input", "Median input"])
                null_method = options
                if options == "Remove":
                    # Removing all rows that contain any null value
                    st.session_state['data'] = st.session_state.data.dropna()
                    st.success("All rows with null values have been removed.")
                elif options == "Mean input":
                    # Filling null values with the mean of each column
                    for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
                        st.session_state.data[column].fillna(st.session_state.data[column].mean(), inplace=True)
                    st.success("Null values have been replaced with the mean of the corresponding column.")
                elif options == "Median input":
                    # Filling null values with the median of each column
                    for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
                        st.session_state.data[column].fillna(st.session_state.data[column].median(), inplace=True)
                    st.success("Null values have been replaced with the median of the corresponding column.")

            if st.checkbox("Scale Data"):
                is_scale = True
                try:
                # Splitting data to train and test
                    X_train, X_test, y_train, y_test = train_test_split(
                        full_data_feature,
                        st.session_state.target_data,
                        test_size=0.25,
                        random_state=555
                    )

                    # Assigning Scaler and encoder Object and fitting the data
                    scaler = MinMaxScaler()
                    encoder = OneHotEncoder()

                    # Fitting anf transforming the data
                    scaled_data_train = scaler.fit_transform(X_train[feature_column_number],
                                                            y_train)

                    encoded_data_train = encoder.fit_transform(
                        X_train[feature_column_text], y_train)
                    
                    st.session_state.preprocessor = ColumnTransformer(
                        transformers = [
                            ('num', scaler, feature_column_number),
                            ('cat', encoder, feature_column_text)
                        ]
                    )

                    st.session_state.preprocessor.fit(X_train, y_train)

                    # Making dataframe out of scaled data train
                    scaled_data_train_df = pd.DataFrame(
                        scaled_data_train, columns=feature_column_number)

                    encoded_data_train_df = pd.DataFrame(
                        encoded_data_train.toarray())

                    full_data_train_scaled_encoded = pd.concat([scaled_data_train_df,
                                                                encoded_data_train_df],
                                                            axis=1)

                    # Transforming data test and make it into dataframe
                    scaled_data_test = scaler.transform(
                        X_test[feature_column_number])

                    encoded_data_test = encoder.transform(
                        X_test[feature_column_text])

                    scaled_data_test_df = pd.DataFrame(
                        scaled_data_test, columns=feature_column_number)

                    encoded_data_test_df = pd.DataFrame(
                        encoded_data_test.toarray())

                    full_data_test_scaled_encoded = pd.concat([scaled_data_test_df,
                                                            encoded_data_test_df],
                                                            axis=1)

                    st.success("The data have been scaled!")

                    # Showing scaled data
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing Scaled Data Train
                    st.markdown("<h4 class='menu-secondary'>Data Train Scaled</h3>",
                                unsafe_allow_html=True)
                    st.write(full_data_train_scaled_encoded)
                    st.write("- The shape of scaled train data :",
                            full_data_train_scaled_encoded.shape)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showin Scaled Data test
                    st.markdown("<h4 class='menu-secondary'>Data Test Scaled</h3>",
                                unsafe_allow_html=True)
                    st.write(full_data_test_scaled_encoded)
                    st.write("- The shape of scaled test data :",
                            full_data_test_scaled_encoded.shape)

                    # Reassing session state to be used later
                    st.session_state['scaled_data_train'] = full_data_train_scaled_encoded
                    st.session_state['scaled_data_test'] = full_data_test_scaled_encoded
                    st.session_state['classification_type'] = classification_type
                    st.session_state['feature_data_train'] = X_train
                    st.session_state['feature_data_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                except Exception as e:
                    st.warning("Cannot scale or preprocess your data, it's may be due to incorrect choose of numerical feature or text feature")
            
            if st.button('Save'):
                if is_scale:
                    df_output_features = {
                        "xtrain":st.session_state['scaled_data_train'],
                        "xtest":st.session_state['scaled_data_test'],
                        "ytrain":st.session_state['y_train'],
                        "ytest":st.session_state['y_test']
                    }
                else:
                    df_output_features = {
                        "ori":st.session_state['data']
                    }
                insert_data(is_scale=is_scale, df_is_null=is_null, null_method=null_method, df_column_number_feature=feature_column_number, 
                            df_column_text_feature=feature_column_text, df_target_column=target_column, df_output_features=df_output_features,
                            dataset_name=st.session_state['data_name'])
                st.success("Data saved into database")

        else:
            st.write("")

# Setting engineering for Regression
    if task_selected == "Feature Engineering for Regression":
        is_scale = False
        is_null = None
        null_method = None
        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Assigning upload file variable
        data_uploader_components(st)

        # Menu if data already stored in session state for classification/regression
        if 'uploaded_file' in st.session_state:
            st.session_state['data'] = st.session_state['uploaded_file']

            pilihan_kolom = list(st.session_state.data.columns)

            # Making column for selecting feature and target
            col1, col2 = st.columns(2)

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            # Assigning option for feature column
            with col1:
                st.markdown("<br>", unsafe_allow_html=True)
                feature_column_number = st.multiselect("Select number column as feature",
                                                       st.session_state.data.columns,
                                                       default=list(
                                                           st.session_state.data.columns),
                                                       placeholder="Select columns")

             # Assigning option for target column
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                target_column = st.selectbox("Select column to be the target",
                                             st.session_state.data.columns)

            col3, col4 = st.columns(2)

            # Assigning option for target column
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                feature_column_text = st.multiselect("Select text column as feature",
                                                     st.session_state.data.columns,
                                                     default=None)

            # Making column for showing features and target
            col5, col6 = st.columns([3, 1])

            full_data_feature = pd.concat([st.session_state.data[feature_column_number],
                                           st.session_state.data[feature_column_text]],
                                          axis=1)

            with col5:
                st.write("List of Feature Data")
                st.write(full_data_feature)
                st.write("- The shape of feature column : ",
                         full_data_feature.shape)

            with col6:
                st.write("Target Data")
                st.write(st.session_state.data[target_column])

            st.session_state['feature_data'] = full_data_feature
            st.session_state['target_data'] = st.session_state.data[target_column]

            # Checking for null values
            if st.checkbox("Check is_null data"):
                # Show if there is null data in the dataset
                null_data = st.session_state.data.isnull().sum()
                st.write("Null values in each column:")
                st.write(null_data)
                is_null = null_data
                # Handling null data options
                options = st.selectbox("Select how to handle null data:",
                                    ["Not handling", "Remove", "Mean input", "Median input"])
                null_method = options
                if options == "Remove":
                    # Removing all rows that contain any null value
                    st.session_state['data'] = st.session_state.data.dropna()
                    st.success("All rows with null values have been removed.")
                elif options == "Mean input":
                    # Filling null values with the mean of each column
                    for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
                        st.session_state.data[column].fillna(st.session_state.data[column].mean(), inplace=True)
                    st.success("Null values have been replaced with the mean of the corresponding column.")
                elif options == "Median input":
                    # Filling null values with the median of each column
                    for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
                        st.session_state.data[column].fillna(st.session_state.data[column].median(), inplace=True)
                    st.success("Null values have been replaced with the median of the corresponding column.")

            if st.checkbox("Scale Data"):
                try:
                    is_scale = True
                    # Splitting data to train and test
                    X_train, X_test, y_train, y_test = train_test_split(
                        full_data_feature,
                        st.session_state.target_data,
                        test_size=0.25,
                        random_state=555
                    )

                    # Assigning Scaler and encoder Object and fitting the data
                    scaler = MinMaxScaler()
                    encoder = OneHotEncoder()

                    # Fitting anf transforming the data
                    scaled_data_train = scaler.fit_transform(X_train[feature_column_number],
                                                            y_train)

                    encoded_data_train = encoder.fit_transform(
                        X_train[feature_column_text], y_train)
                    
                    st.session_state.preprocessor = ColumnTransformer(
                        transformers = [
                            ('num', scaler, feature_column_number),
                            ('cat', encoder, feature_column_text)
                        ]
                    )

                    st.session_state.preprocessor.fit(X_train, y_train)

                    # Making dataframe out of scaled data train
                    scaled_data_train_df = pd.DataFrame(
                        scaled_data_train, columns=feature_column_number)

                    encoded_data_train_df = pd.DataFrame(
                        encoded_data_train.toarray())

                    full_data_train_scaled_encoded = pd.concat([scaled_data_train_df,
                                                                encoded_data_train_df],
                                                            axis=1)

                    # Transforming data test and make it into dataframe
                    scaled_data_test = scaler.transform(
                        X_test[feature_column_number])

                    encoded_data_test = encoder.transform(
                        X_test[feature_column_text])

                    scaled_data_test_df = pd.DataFrame(
                        scaled_data_test, columns=feature_column_number)

                    encoded_data_test_df = pd.DataFrame(
                        encoded_data_test.toarray())

                    full_data_test_scaled_encoded = pd.concat([scaled_data_test_df,
                                                            encoded_data_test_df],
                                                            axis=1)

                    st.success("The data have been scaled!")

                    # Showing scaled data
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing Scaled Data Train
                    st.markdown("<h4 class='menu-secondary'>Data Train Scaled</h3>",
                                unsafe_allow_html=True)
                    st.write(full_data_train_scaled_encoded)
                    st.write("- The shape of scaled train data :",
                            full_data_train_scaled_encoded.shape)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showin Scaled Data test
                    st.markdown("<h4 class='menu-secondary'>Data Test Scaled</h3>",
                                unsafe_allow_html=True)
                    st.write(full_data_test_scaled_encoded)
                    st.write("- The shape of scaled test data :",
                            full_data_test_scaled_encoded.shape)

                    # Reassing session state to be used later
                    st.session_state['scaled_data_train'] = full_data_train_scaled_encoded
                    st.session_state['scaled_data_test'] = full_data_test_scaled_encoded
                    st.session_state['feature_data_train'] = X_train
                    st.session_state['feature_data_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                except:
                    st.warning("Cannot scale or preprocess your data, it's may be due to incorrect choose of numerical feature or text feature")
            if st.button('Save'):
                if is_scale:
                    df_output_features = {
                        "xtrain":st.session_state['scaled_data_train'],
                        "xtest":st.session_state['scaled_data_test'],
                        "ytrain":st.session_state['y_train'],
                        "ytest":st.session_state['y_test']
                    }
                else:
                    df_output_features = {
                        "ori":st.session_state['data']
                    }
                insert_data(is_scale=is_scale, df_is_null=is_null, null_method=null_method, df_column_number_feature=feature_column_number, 
                            df_column_text_feature=feature_column_text, df_target_column=target_column, df_output_features=df_output_features,
                            dataset_name=st.session_state['data_name'])
                st.success("Data saved into database")
        else:
            st.write("")

    # Option feature engineering for clustering
    if task_selected == "Feature Engineering for Clustering":
        is_scale = False
        is_null = None
        null_method = None
        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Assigning upload file variable
        data_uploader_components(st)

        # Menu if data already stored in session state for clustering
        if 'uploaded_file' in st.session_state:
            st.session_state['data'] = st.session_state['uploaded_file']

            pilihan_kolom = list(st.session_state.data.columns)

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            # Option menu of feature column for clustering
            with col1:
                feature_column_number = st.multiselect("Select number column to be featured for Clustering",
                                                       st.session_state.data.columns,
                                                       default=list(
                                                           st.session_state.data.columns),
                                                       placeholder="Select columns")

            with col2:
                feature_column_text = st.multiselect("Select text column to be featured for Clustering",
                                                     st.session_state.data.columns,
                                                     default=None,
                                                     placeholder="Select columns")

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            feature_data = pd.concat([st.session_state.data[feature_column_number],
                                      st.session_state.data[feature_column_text]],
                                     axis=1)

            st.write(feature_data)
            st.write("- The shape of the data :",
                     feature_data.shape)

            # st.session_state['feature_data'] = feature_data

            # Checking for null values
            if st.checkbox("Check is_null data"):
                # Show if there is null data in the dataset
                null_data = st.session_state.data.isnull().sum()
                st.write("Null values in each column:")
                st.write(null_data)
                is_null = null_data
                # Handling null data options
                options = st.selectbox("Select how to handle null data:",
                                    ["Not handling", "Remove", "Mean input", "Median input"])
                null_method = options
                if options == "Remove":
                    # Removing all rows that contain any null value
                    st.session_state['data'] = st.session_state.data.dropna()
                    st.success("All rows with null values have been removed.")
                elif options == "Mean input":
                    # Filling null values with the mean of each column
                    for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
                        st.session_state.data[column].fillna(st.session_state.data[column].mean(), inplace=True)
                    st.success("Null values have been replaced with the mean of the corresponding column.")
                elif options == "Median input":
                    # Filling null values with the median of each column
                    for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
                        st.session_state.data[column].fillna(st.session_state.data[column].median(), inplace=True)
                    st.success("Null values have been replaced with the median of the corresponding column.")

            if st.checkbox("Scale Data"):
                try:
                    is_scale = True
                    # Initiate scaler object
                    scaler = MinMaxScaler()
                    encoder = OneHotEncoder()

                    # Fitting and transforming the data
                    scaled_data_train_df = pd.DataFrame(scaler.fit_transform(
                        feature_data[feature_column_number]), columns=feature_column_number)

                    encoded_data_train_df = pd.DataFrame(encoder.fit_transform(
                        feature_data[feature_column_text]).toarray())
                    
                    st.session_state.preprocessor = ColumnTransformer(
                        transformers = [
                            ('num', scaler, feature_column_number),
                            ('cat', encoder, feature_column_text)
                        ]
                    )

                    st.session_state.preprocessor.fit(feature_data)

                    full_data_train_scaled_encoded = pd.concat([scaled_data_train_df,
                                                                encoded_data_train_df],
                                                            axis=1)

                    st.success("The data have been scaled!")

                    # Giving one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing scaled data train
                    st.markdown("<h4 class='menu-secondary'>Data Train Scaled</h3>",
                                unsafe_allow_html=True)
                    st.write(full_data_train_scaled_encoded)
                    st.write("- The shape of scaled train data :",
                            full_data_train_scaled_encoded.shape)

                    # Saving scaled data train into session state
                    st.session_state.scaled_data_train = full_data_train_scaled_encoded
                except Exception as e:
                    st.warning("Cannot scale or preprocess your data, it's may be due to incorrect choose of numerical feature or text feature")

            if st.button('Save'):
                if is_scale:
                    df_output_features = {
                        "train":st.session_state['scaled_data_train'],
                    }
                else:
                    df_output_features = {
                        "ori":st.session_state['data']
                    }
                insert_data(is_scale=is_scale, df_is_null=is_null, null_method=null_method, df_column_number_feature=feature_column_number, 
                            df_column_text_feature=feature_column_text, df_target_column=None, df_output_features=df_output_features,
                            dataset_name=st.session_state['data_name'])
                st.success("Data saved into database")