import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from util import save_result, show_roc_auc_score_binary_class
from util import show_roc_auc_score_multi_class, plot_confusion_matrix_multi_class
import datetime

from streamlit_option_menu import option_menu
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR, SVC

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score
from util import config, generate_random_alphanumeric, add_index_column_if_first_is_float, transform_digits, remove_punctuation, dtype_to_sql
from sqlalchemy import create_engine

def save_data(df, dataset_name, model_description, data_to_save, task_selected, model_pipeline):
    save_result(data_to_save, task_selected, model_pipeline)
    # try:
    insert_data(df, dataset_name, model_description)
    # except:
    #     st.warning("Database is not connected please check is your database is on, or reload the page")

def insert_data(df, dataset_name, model_description):
    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()

    # Generate a unique identifier
    uid = generate_random_alphanumeric()
    uid = str(uid).replace("-","")

    # Construct table name
    table_name = f"predictions_{uid}_{dataset_name.replace('.csv','')}"

    valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
    df = df[valid_columns]
    df = add_index_column_if_first_is_float(df)
    
    # Create a new table based on the structure of df
    cols_with_types = ", ".join([f"{transform_digits(remove_punctuation(str(col)).replace(' ','_').lower())} {dtype_to_sql(df[col].dtype.name)}" for col in df.columns if not col.startswith('Unnamed')])
    create_table_query = f"CREATE TABLE {table_name} ({cols_with_types})"
    connection.execute(create_table_query)

    # Prepare and execute the insertion query for the main predictions_master table
    connection.execute(
        "INSERT INTO predictions_master (id, dataset, model_prediction, prediction_at) VALUES (%s, %s, %s, %s)",
        (str(uid), dataset_name, model_description, datetime.datetime.now())
    )

    # Insert DataFrame data into the newly created table using psycopg2's execute_values for bulk insertion
    # Filter out columns starting with 'Unnamed'
    df = df[[col for col in df.columns if not str(col).startswith('Unnamed')]]

    # Transform column names
    df.columns = [transform_digits(remove_punctuation(col).replace(' ','_').lower()) for col in df.columns]
    df.to_sql(table_name, con=engine, index=False, if_exists='append', method='multi')
    
    print("Data input successfully")
    st.success(f"Data saved into database with table name: {table_name}")

def show_data_prediction(data_train_full_prediction, data_test_full_prediction=None):
    st.markdown("<h4 class='menu-secondary'>Train Data with Prediction</h4>",
                                unsafe_allow_html=True)
    st.write(data_train_full_prediction)
    st.write("- The shape of train data with prediction : ",
                data_train_full_prediction.shape)
    # Adding one space
    if data_test_full_prediction is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 class='menu-secondary'>Test Data with Prediction</h4>",
                    unsafe_allow_html=True)
        st.write(data_test_full_prediction)
        st.write("- The shape of test data with prediction : ",
                    data_test_full_prediction.shape)
    
def show_classification_report(classification_report_train, classification_report_test):
    st.markdown("<h4 class='menu-secondary'>Train Score</h4>", unsafe_allow_html=True)
    st.write(classification_report_train, help="A classification report shows key metrics in assessing the performance of a classification model. This includes precision, recall, f1-score, and support for each class.")
    
    st.markdown("<h4 class='menu-secondary'>Test Score</h4>", unsafe_allow_html=True)
    st.write(classification_report_test, help="A classification report shows key metrics in assessing the performance of a classification model. This includes precision, recall, f1-score, and support for each class.")

    # Additional explanations for the metrics
    st.markdown("### Metric Definitions:")
    st.write("**Precision:** Precision is the ratio of correctly predicted positive observations to the total predicted positives. It is a measure of a classifier's exactness. High precision relates to the low false positive rate.")
    st.write("**Recall:** Recall is the ratio of correctly predicted positive observations to the all observations in actual class. It is a measure of a classifier's completeness. High recall indicates that the class is correctly recognized (low false negative rate).")
    st.write("**F1 Score:** The F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is particularly useful when the classes are imbalanced. The F1 score is the harmonic mean of Precision and Recall.")

import streamlit as st

def show_roc_auc_report(roc_auc_train, roc_auc_test):
    st.markdown("<h5 class='menu-secondary'>Train ROC-AUC Score</h5>", unsafe_allow_html=True)
    st.write(roc_auc_train)
    st.markdown("<h5 class='menu-secondary'>Test ROC-AUC Score</h5>", unsafe_allow_html=True)
    st.write(roc_auc_test)

    # Explanation for ROC-AUC
    st.markdown("### ROC-AUC Explanation:")
    st.write("""
    **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** This metric is used to measure the effectiveness of a classification model at distinguishing between classes. The ROC is a probability curve that plots the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher AUC values indicate better model performance.
    """)

def show_confussion_matrix(cm, label_target):
    st.markdown("<h4 class='menu-secondary'>Confusion Matrix Score</h4>", unsafe_allow_html=True)

    # Showing Confusion Matrix Display
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_target)
    disp.plot()
    st.pyplot(plt)

    # Explanation for Confusion Matrix
    st.markdown("### Confusion Matrix Explanation:")
    st.write("""
    The Confusion Matrix is a visualization tool primarily used to show the performance of a classification model. The matrix displays the actual labels vs. the labels predicted by the model, allowing users to identify how many predictions were correctly or incorrectly made for each class. Here's what each term represents:
    - **True Positives (TP):** The number of correct predictions that an instance is positive.
    - **True Negatives (TN):** The number of correct predictions that an instance is negative.
    - **False Positives (FP):** The number of incorrect predictions that an instance is positive (also known as Type I error).
    - **False Negatives (FN):** The number of incorrect predictions that an instance is negative (also known as Type II error).
    Understanding the distribution of these values can help diagnose the performance characteristics of the model, especially in terms of its precision, recall, and overall accuracy.
    """)

def show_classification_metrics(data_train_full_prediction, data_test_full_prediction,
                           classification_report_train, classification_report_test,
                           roc_auc_train, roc_auc_test, cm, label_target):
    with st.expander("Show Data"):
        show_data_prediction(data_train_full_prediction, data_test_full_prediction)

    # Giving space
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing score
    with st.expander("Show Classification Score"):
        show_classification_report(classification_report_train, classification_report_test)

    # Giving two spaces
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing ROC-AUC Score
    with st.expander("Show ROC-AUC Report"):
        show_roc_auc_report(roc_auc_train, roc_auc_test)

    # Giving two spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing Confusion Matrix
    with st.expander("Show Confusion Matrix"):
        show_confussion_matrix(cm, label_target)

def show_regression_metrics(data_train_full_prediction, data_test_full_prediction, mae_train, mae_test, mse_train, mse_test):
    with st.expander("Show Data"):
        show_data_prediction(data_train_full_prediction, data_test_full_prediction)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing mean absolute error
    with st.expander("Show Mean Absolute Error"):
        st.write("Train Score")
        st.write(mae_train)
        st.write("Test Score")
        st.write(mae_test)
        # Explanation for MAE
        st.markdown("### Mean Absolute Error (MAE) Explanation:")
        st.write("""
        Mean Absolute Error (MAE) is a measure of errors between paired observations expressing the same phenomenon. Comparing predictions and outcomes, MAE is the average of the absolute differences between the predicted values and actual values without considering the direction. It's a linear score which means all individual differences are weighted equally. It's particularly useful in regression and forecasting models.
        """)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    # Showing mean squared error
    with st.expander("Show Mean Squared Error"):
        st.write("Train Score")
        st.write(mse_train)
        st.write("Test Score")
        st.write(mse_test)
        # Explanation for MSE
        st.markdown("### Mean Squared Error (MSE) Explanation:")
        st.write("""
        Mean Squared Error (MSE) is a risk metric that calculates the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. MSE gives a higher weight to larger errors due to the squaring of each term, which can be particularly important in real-world contexts where large errors are particularly undesirable.
        """)

    # Giving three spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def show_clustering_metrics(data_full_clustered, calinski_harabasz, davies_bouldin):
    with st.expander("Show Data"):
        show_data_prediction(data_full_clustered)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Show Evaluation Score"):

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        st.write("Calinski-Harabasz Index")
        st.write(calinski_harabasz)
        # Explanation for Calinski-Harabasz Index
        st.markdown("### Calinski-Harabasz Index Explanation:")
        st.write("""
        The Calinski-Harabasz Index, also known as the CH Score, measures the quality of clustering. It is calculated as the ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters. The higher the CH Score, the better the clustering performance, indicating clusters are well-separated and dense.
        """)

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        st.write("Davies-Bouldin Index")
        st.write(davies_bouldin)
        # Explanation for Davies-Bouldin Index
        st.markdown("### Davies-Bouldin Index Explanation:")
        st.write("""
        The Davies-Bouldin Index is a measure of clustering validation. It evaluates clustering algorithms by calculating the average 'similarity' between each cluster and its most similar cluster, where similarity is a measure that compares the distance between clusters with the size of the clusters themselves. Lower values of the index indicate better clustering performance, reflecting clusters with less overlap.
        """)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

def modeling_page(st):
    st.markdown("<h2 class='menu-title'>Modelling</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Designing machine learning model alghorithm and its hyper-parameters</h6>",
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

    try:
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
    except:
        st.warning("Database is not connected please check is your database is on, or reload the page")

    if task_selected == "Classification":
        st.write("Classification is a Machine Learning task where the model learns from the data to assign a category label to each data point.")

    elif task_selected == "Regression":
        st.write("Regression is a type of Machine Learning that estimates relationships among variables. It is used when the target, or dependent variable, is continuous.")

    elif task_selected == "Clustering":
        st.write("Clustering is a Machine Learning technique that involves grouping a set of objects in such a way that objects in the same group (a cluster) are more similar to each other than to those in other groups.")

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    # Configuring Classification Task
    data_to_save = {"Experimentation": task_selected}
    if task_selected == "Classification":
        is_train = False
        # Assigning scaled_data_train key
        if "scaled_data_train" not in st.session_state:
            st.session_state['scaled_data_train'] = ""

        # Checking if scaled_data_train is DataFrame
        if type(st.session_state.scaled_data_train) == pd.DataFrame:
            st.markdown("<h3 class='menu-secondary'>Feature Data</h3>",
                        unsafe_allow_html=True)
            st.write(st.session_state.scaled_data_train)
            st.write(" - Data Shape :",
                     st.session_state.scaled_data_train.shape)
        else:
            print("")
            # Setting the upload variabel
            # uploaded_file = st.file_uploader("Choose a file to upload for training data",
            #                                  type="csv",
            #                                  help="The file will be used for training the Machine Learning",
            #                                  )

            # # Setting the upload options when there's file on uploader menu
            # if uploaded_file is not None:
            #     try:
            #         # Uploading Dataframe
            #         dataframe = get_data(uploaded_file)

            #         X = dataframe.drop(columns="Outcome")
            #         y = dataframe["Outcome"]

            #         # Storing dataframe to session state
            #         if 'X' not in st.session_state:
            #             st.session_state["X"] = X

            #         if 'y' not in st.session_state:
            #             st.session_state["y"] = y

            #     except:
            #         st.markdown("<span class='info-box'>Please upload any data</span>",
            #                     unsafe_allow_html=True)

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-secondary'>Model Configuration</h3>",
                    unsafe_allow_html=True)

        # Selecting Model for Classification
        model_selection = st.selectbox(
            "Select Machine Learning Model for Classification Task",
            ("Logistic Regression", "Random Forest", "SVM")
        )
        data_to_save['model_selection'] = model_selection
        st.write("Model selected:", model_selection)

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

            
        # Setting Logistic Regression Model
        if model_selection == "Logistic Regression":
            col1, col2, col3 = st.columns(3)

            with col1:
                # Setting Logistic Regression Penalty
                log_res_penalty = st.radio(
                    "Norm of the penalty",
                    ('l2', 'l1', 'none'))

            with col2:
                # Setting Logistis Regression Solver
                log_res_solver = st.radio(
                    "Algorithm optimization",
                    ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
                     ))

            with col3:
                # Inverse of regularization strength
                log_res_inverse = st.number_input(
                    "Inverse of regularization",
                    min_value=0.001,
                    value=1.0,
                    step=0.01)

            data_to_save['configuration'] = {
                "log_res_penalty" : log_res_penalty,
                "log_res_solver" : log_res_solver,
                "log_res_inverse" : log_res_inverse
            }
            # Logistic Regression Object
            log_res_obj = LogisticRegression(
                penalty=log_res_penalty, C=log_res_inverse, solver=log_res_solver)

            # Fitting Data to Logistic Regression Model
            # try:
            if st.button("Fit Data to Logistic Model"):
                experiment_date = str(datetime.datetime.now())
                data_to_save['experiment_date'] = experiment_date
                # Initiating variable to fir data
                X_train = st.session_state.scaled_data_train
                X_test = st.session_state.scaled_data_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                # Fitting model to data
                log_res_obj.fit(X_train, y_train)
                is_train = True
                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success("Training Success")

                # Predicting train data
                y_train_predict = log_res_obj.predict(X_train)
                y_train_predict_df = pd.DataFrame(y_train_predict)
                y_train_predict_proba = pd.DataFrame(
                    log_res_obj.predict_proba(X_train))

                # Predicting test data
                y_test_predict = log_res_obj.predict(X_test)
                y_test_predict_df = pd.DataFrame(y_test_predict)
                y_test_predict_proba = pd.DataFrame(
                    log_res_obj.predict_proba(X_test))

                label_target = list(y_train.unique())

                # Predicting F1 score
                classification_report_train = pd.DataFrame(
                    classification_report(
                        y_train,
                        y_train_predict,
                        labels=label_target,
                        output_dict=True
                    ))
                classification_report_test = pd.DataFrame(
                    classification_report(
                        y_test,
                        y_test_predict,
                        labels=label_target,
                        output_dict=True
                    ))
                
                if st.session_state['classification_type'] == 'Binary Class':
                    roc_auc_train = show_roc_auc_score_binary_class(
                            y_train, y_train_predict_proba)
                    roc_auc_test = show_roc_auc_score_binary_class(
                        y_test, y_test_predict_proba)
                else:
                    roc_auc_train = show_roc_auc_score_multi_class(
                            y_train, y_train_predict_proba)
                    roc_auc_test = show_roc_auc_score_multi_class(
                        y_test, y_test_predict_proba)

                cm = confusion_matrix(
                        y_test, y_test_predict, labels=label_target)

                # Showing Data Real vs Prediction
                # Changing target series column to dataframe
                y_train_df = y_train.to_frame()
                y_test_df = y_test.to_frame()

                # Concatting target actual column
                target_actual_full = pd.concat([y_train_df, y_test_df],
                                                axis=0)

                # Adding index to predicttion of target train
                train_index = list(y_train_df.index)
                y_train_predict_df['index'] = train_index
                y_train_predict_df.set_index('index', inplace=True)

                # Adding index to prediction of target test
                test_index = list(y_test_df.index)
                y_test_predict_df['index'] = test_index
                y_test_predict_df.set_index('index', inplace=True)

                # Renaming target columns name of train data
                y_train_df.columns = ["Target Actual"]
                y_train_predict_df.columns = ["Target Predicted"]

                # Showing data train full
                data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                        y_train_df, y_train_predict_df],
                                                        axis=1)
                # Renaming target columns name of test data
                y_test_df.columns = ["Target Actual"]
                y_test_predict_df.columns = ["Target Predicted"]

                # Showing data train full
                data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                        y_test_df, y_test_predict_df],
                                                        axis=1)
                ## show classification metrics components
                show_classification_metrics(data_train_full_prediction, data_test_full_prediction,
                        classification_report_train, classification_report_test,
                        roc_auc_train, roc_auc_test, cm, label_target)
                    
                data_to_save['outputs'] = {
                    "y_train_predict": data_train_full_prediction,
                    "y_test_predict": data_test_full_prediction,
                    "classification_score_train": classification_report_train,
                    "classification_score_test": classification_report_test,
                    "roc_auc_train": roc_auc_train,
                    "roc_auc_test": roc_auc_test,
                    "confussion_matrix":cm
                }
                st.session_state.data_to_save = data_to_save

                log_res_pipeline = Pipeline(steps=[
                    ('preprocessor', st.session_state.preprocessor),
                    ('classifier', log_res_obj)
                ])
                save_data(pd.concat([data_train_full_prediction, data_test_full_prediction]), st.session_state['data_name'], model_selection, data_to_save, task_selected, log_res_pipeline)
            # except Exception as e:
            #     st.warning("Cannot train your data, please upload your data and scale your data to train the model")
                    
        # Setting Random Forest Classifier Model
        if model_selection == "Random Forest":

            col1, col2, col3 = st.columns(3)

            with col1:
                # Setting Random Forest Classifier Split Criterion
                rfc_criterion = st.radio(
                    "Split Criterion",
                    ('gini', 'entropy', 'log_loss'))

            with col2:
                # Minimal Sample Split
                rfc_max_depth = st.number_input(
                    "Maximum Depth of the Tree",
                    min_value=2,
                    value=100,
                    step=1)

            with col3:
                # Minimum number of samples to be at a left node
                rfc_min_samples_leaf = st.number_input(
                    "Minium Sample Leaf",
                    min_value=2,
                    step=1)
            data_to_save['configuration'] = {
                "rfc_criterion" : rfc_criterion,
                "rfc_max_depth" : rfc_max_depth,
                "rfc_min_samples_leaf" : rfc_min_samples_leaf
            }
            # Random Forest Classifier Object
            rfc_obj = RandomForestClassifier(
                criterion=rfc_criterion,
                max_depth=rfc_max_depth,
                min_samples_leaf=rfc_min_samples_leaf)

            # Fitting Data to Random Forest Classifier Model
            if st.button("Fit Data to Random Forest Model"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to fit data
                    X_train = st.session_state.scaled_data_train
                    X_test = st.session_state.scaled_data_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    # Fitting model to data
                    rfc_obj.fit(X_train, y_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Predicting train data
                    y_train_predict = rfc_obj.predict(X_train)
                    y_train_predict_df = pd.DataFrame(y_train_predict)
                    y_train_predict_proba = pd.DataFrame(
                        rfc_obj.predict_proba(X_train))

                    # Predicting test data
                    y_test_predict = rfc_obj.predict(X_test)
                    y_test_predict_df = pd.DataFrame(y_test_predict)
                    y_test_predict_proba = pd.DataFrame(
                        rfc_obj.predict_proba(X_test))

                    label_target = list(y_train.unique())

                    # Predicting F1 score for train data
                    classification_report_train = pd.DataFrame(classification_report(
                        y_train,
                        y_train_predict,
                        labels=label_target,
                        output_dict=True
                    ))

                    # Predicting F1 score for test data
                    classification_report_test = pd.DataFrame(classification_report(
                        y_test,
                        y_test_predict,
                        labels=label_target,
                        output_dict=True
                    ))

                    if st.session_state['classification_type'] == 'Binary Class':
                        roc_auc_train = show_roc_auc_score_binary_class(
                                y_train, y_train_predict_proba)
                        roc_auc_test = show_roc_auc_score_binary_class(
                            y_test, y_test_predict_proba)
                    else:
                        roc_auc_train = show_roc_auc_score_multi_class(
                                y_train, y_train_predict_proba)
                        roc_auc_test = show_roc_auc_score_multi_class(
                            y_test, y_test_predict_proba)

                    cm = confusion_matrix(
                            y_test, y_test_predict, labels=label_target)

                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                    axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    label_target = list(y_train.unique())

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                            axis=1)
                    
                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                            y_test_df, y_test_predict_df],
                                                            axis=1)

                    show_classification_metrics(data_train_full_prediction, data_test_full_prediction,
                            classification_report_train, classification_report_test,
                            roc_auc_train, roc_auc_test, cm, label_target)
                    
                    data_to_save['outputs'] = {
                        "y_train_predict": data_train_full_prediction,
                        "y_test_predict": data_test_full_prediction,
                        "classification_score_train": classification_report_train,
                        "classification_score_test": classification_report_test,
                        "roc_auc_train": roc_auc_train,
                        "roc_auc_test": roc_auc_test,
                        "confussion_matrix":cm
                    }
                    st.session_state.data_to_save = data_to_save

                    rfc_pipeline = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('classifier', rfc_obj)
                    ])

                    save_data(pd.concat([data_train_full_prediction, data_test_full_prediction]), st.session_state['data_name'], model_selection, data_to_save, task_selected, rfc_pipeline)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")

        # Setting SVC Model
        if model_selection == "SVM":

            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7, col8 = st.columns(4)

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            with col1:
                kernel = st.radio(
                    "Kernel type",
                    ("rbf", "linear", "poly", "sigmoid", "precomputed")
                )

            with col2:
                degree = st.number_input(
                    "Degree of Polynomial kernel function",
                    value=3,
                    step=1,
                    min_value=1
                )

            with col3:
                gamma = st.radio(
                    "Kernel coefficient for 'rbf', 'poly', & 'sigmoid'",
                    ("scale", "auto")
                )

            with col4:
                tol = st.number_input(
                    "Tolerance for stopping criterion",
                    value=0.001,
                    step=0.001,
                    max_value=1.0,
                    min_value=0.00001
                )

            with col5:
                C = st.number_input(
                    "Regularization parameter",
                    value=1.0,
                    min_value=0.1,
                    step=0.1
                )

            with col7:
                shrinking = st.radio(
                    "Shrinking heuristic",
                    (True, False)
                )

            with col8:
                cache_size = int(400)

            data_to_save['configuration'] = {
                "kernel":kernel,
                "degree":degree,
                "gamma":gamma,
                "tol":tol,
                "C":C,
                "shrinking":shrinking,
                "cache_size":cache_size
            }

            # Logistic Regression Object
            svc_obj = SVC(
                C=C, kernel=kernel,
                degree=degree,
                gamma=gamma,
                shrinking=shrinking,
                cache_size=cache_size,
                tol=tol,
                probability=True
            )

            # Fitting Data to Logistic Regression Model
            if st.button("Fit Data to SVM Model"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to fir data
                    X_train = st.session_state.scaled_data_train
                    X_test = st.session_state.scaled_data_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    # Fitting model to data
                    svc_obj.fit(X_train, y_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Predicting train data
                    y_train_predict = svc_obj.predict(X_train)
                    y_train_predict_df = pd.DataFrame(y_train_predict)
                    y_train_predict_proba = pd.DataFrame(
                        svc_obj.predict_proba(X_train))

                    # Predicting test data
                    y_test_predict = svc_obj.predict(X_test)
                    y_test_predict_df = pd.DataFrame(y_test_predict)
                    y_test_predict_proba = pd.DataFrame(
                        svc_obj.predict_proba(X_test))

                    # creating label target

                    label_target = list(y_train.unique())

                    # Predicting F1 score
                    classification_report_train = pd.DataFrame(
                        classification_report(
                            y_train,
                            y_train_predict,
                            labels=label_target,
                            output_dict=True
                        ))
                    classification_report_test = pd.DataFrame(
                        classification_report(
                            y_test,
                            y_test_predict,
                            labels=label_target,
                            output_dict=True
                        ))
                    
                    if st.session_state['classification_type'] == 'Binary Class':
                        roc_auc_train = show_roc_auc_score_binary_class(
                                y_train, y_train_predict_proba)
                        roc_auc_test = show_roc_auc_score_binary_class(
                            y_test, y_test_predict_proba)
                    else:
                        roc_auc_train = show_roc_auc_score_multi_class(
                                y_train, y_train_predict_proba)
                        roc_auc_test = show_roc_auc_score_multi_class(
                            y_test, y_test_predict_proba)

                    cm = confusion_matrix(
                            y_test, y_test_predict, labels=label_target)

                    # Showing Data Real vs Prediction
                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                    axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                            axis=1)
                    
                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                            y_test_df, y_test_predict_df],
                                                            axis=1)

                    show_classification_metrics(data_train_full_prediction, data_test_full_prediction,
                            classification_report_train, classification_report_test,
                            roc_auc_train, roc_auc_test, cm, label_target)

                    data_to_save['outputs'] = {
                        "y_train_predict": data_train_full_prediction,
                        "y_test_predict": data_test_full_prediction,
                        "classification_score_train": classification_report_train,
                        "classification_score_test": classification_report_test,
                        "roc_auc_train": roc_auc_train,
                        "roc_auc_test": roc_auc_test,
                        "confussion_matrix":cm
                    }
                    st.session_state.data_to_save = data_to_save

                    svc_pipeline = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('classifier', svc_obj)
                    ])

                    save_data(pd.concat([data_train_full_prediction, data_test_full_prediction]), st.session_state['data_name'], model_selection, data_to_save, task_selected, svc_pipeline)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")
    # Configuring Regression Task
    if task_selected == "Regression":

        # Assigning scaled_data_train key
        if "scaled_data_train" not in st.session_state:
            st.session_state['scaled_data_train'] = ""

        # Checking if scaled_data_train is DataFrame
        if type(st.session_state.scaled_data_train) == pd.DataFrame:
            st.markdown("<h3 class='menu-secondary'>Feature Data</h3>",
                    unsafe_allow_html=True)
            st.write(st.session_state.scaled_data_train)
            st.write(" - Data Shape :",
                     st.session_state.scaled_data_train.shape)
        else:
            print("")
            # # Setting the upload variabel
            # uploaded_file = st.file_uploader("Choose a file to upload for training data",
            #                                  type="csv",
            #                                  help="The file will be used for training the Machine Learning",
            #                                  )

            # # Setting the upload options when there's file on uploader menu
            # if uploaded_file is not None:
            #     try:
            #         # Uploading Dataframe
            #         dataframe = get_data(uploaded_file)

            #         X = dataframe.drop(columns="Outcome")
            #         y = dataframe["Outcome"]

            #         # Storing feature dataframe to session state
            #         if 'X' not in st.session_state:
            #             st.session_state["X"] = X

            #         # Storing target series to session state
            #         if 'y' not in st.session_state:
            #             st.session_state["y"] = y

            #     except:
            #         st.markdown("<span class='info-box'>Please upload any data</span>",
            #                     unsafe_allow_html=True)

        # Markdown to give space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-secondary'>Model Configuration</h3>",
                    unsafe_allow_html=True)

        # Showing option of the Model
        model_selection = st.selectbox(
            "Select Machine Learning Model for Regression Task",
            ("Linear Regression", "Random Forest", "SVM")
        )
        st.write("Model selected:", model_selection)
        data_to_save['model_selection'] = model_selection
        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Option if Linear Regression selected
        if model_selection == 'Linear Regression':

            col1, col2 = st.columns(2)

            with col1:
                # Setting Linear Regression fitting intercept
                lin_reg_fit_intercept = st.radio(
                    "Calculating the intercept for the model",
                    (True, False),
                    horizontal=True
                )

            with col2:
                # Setting Linear Regression positive coefficients
                lin_reg_positive = st.radio(
                    "Forcing the coefficients to be positive",
                    (False, True),
                    horizontal=True
                )
            data_to_save['configuration']={
                "lin_reg_fit_intercept":lin_reg_fit_intercept,
                "lin_reg_positive":lin_reg_positive
            }

            # Linear Regression Object
            lin_reg_obj = LinearRegression(
                fit_intercept=lin_reg_fit_intercept,
                positive=lin_reg_positive
            )

            # Fitting Data to Logistic Regression Model
            if st.button("Fit Data to Linear Regression Model"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to data fitting
                    X_train = st.session_state.scaled_data_train
                    X_test = st.session_state.scaled_data_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    # Fitting model to data
                    lin_reg_obj.fit(X_train, y_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Predicting train data
                    y_train_predict = lin_reg_obj.predict(X_train)
                    y_train_predict_df = pd.DataFrame(y_train_predict)

                    # Predicting test data
                    y_test_predict = lin_reg_obj.predict(X_test)
                    y_test_predict_df = pd.DataFrame(y_test_predict)

                    # Calculating mean absolute error
                    mae_train = mean_absolute_error(
                        y_train, y_train_predict)
                    mae_test = mean_absolute_error(
                        y_test, y_test_predict)

                    # Calculating mean squarred error
                    mse_train = mean_squared_error(
                        y_train, y_train_predict)
                    mse_test = mean_squared_error(
                    y_test, y_test_predict)

                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                    axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                            axis=1)
                    
                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                            y_test_df, y_test_predict_df],
                                                            axis=1)
                    data_to_save['outputs'] = {
                        "y_train_predict" : data_train_full_prediction,
                        "y_test_predict" : data_test_full_prediction,
                        "mse_train": mse_train,
                        "mse_test":mse_test,
                        "mae_train":mae_train,
                        "mae_test":mae_test
                    }
                    show_regression_metrics(data_train_full_prediction, data_test_full_prediction, mae_train, mae_test, mse_train, mse_test)

                    lin_reg_pipeline = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('regressor', lin_reg_obj)
                    ])

                    save_data(pd.concat([data_train_full_prediction, data_test_full_prediction]), st.session_state['data_name'], model_selection, data_to_save, task_selected, lin_reg_pipeline)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")

        # Option if Random Forest Regressor selected
        if model_selection == 'Random Forest':

            col1, col2, col3 = st.columns(3)

            with col1:
                # Setting Random Forest Classifier Split Criterion
                rfr_criterion = st.radio(
                    "Split Criterion",
                    ('squared_error', 'absolute_error',
                     'friedman_mse', 'poisson'))

            with col2:
                # Minimal Sample Split
                rfr_max_depth = st.number_input(
                    "Maximum Depth of the Tree",
                    min_value=2,
                    value=100,
                    step=1)

            with col3:
                # Minimum number of samples to be at a left node
                rfr_min_samples_leaf = st.number_input(
                    "Minium Sample Leaf",
                    min_value=2,
                    step=1)
            data_to_save["configuration"]={
                "rfr_max_depth":rfr_max_depth,
                "rfr_min_samples_leaf": rfr_min_samples_leaf,
                "rfr_criterion":rfr_criterion
            }
            # Random Forest Regressor Object
            rfr_obj = RandomForestRegressor(
                criterion=rfr_criterion,
                max_depth=rfr_max_depth,
                min_samples_leaf=rfr_min_samples_leaf)

            # Fitting Data to Random Forest Regressor Model
            if st.button("Fit Data to Random Forest Model"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to data fitting
                    X_train = st.session_state.scaled_data_train
                    X_test = st.session_state.scaled_data_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    # Fitting model to data
                    rfr_obj.fit(X_train, y_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Predicting train data
                    y_train_predict = rfr_obj.predict(X_train)
                    y_train_predict_df = pd.DataFrame(y_train_predict)

                    # Predicting test data
                    y_test_predict = rfr_obj.predict(X_test)
                    y_test_predict_df = pd.DataFrame(y_test_predict)

                    # Calculating mean absolute error
                    mae_train = mean_absolute_error(
                        y_train, y_train_predict)
                    mae_test = mean_absolute_error(
                        y_test, y_test_predict)

                    # Calculating mean squarred error
                    mse_train = mean_squared_error(
                        y_train, y_train_predict)
                    mse_test = mean_squared_error(
                        y_test, y_test_predict)

                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                    axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                            axis=1)
                    
                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                            y_test_df, y_test_predict_df],
                                                            axis=1)
                    data_to_save['outputs'] = {
                        "y_train_predict" : data_train_full_prediction,
                        "y_test_predict" : data_test_full_prediction,
                        "mse_train": mse_train,
                        "mse_test":mse_test,
                        "mae_train":mae_train,
                        "mae_test":mae_test
                    }
                    show_regression_metrics(data_train_full_prediction, data_test_full_prediction, mae_train, mae_test, mse_train, mse_test)

                    rfr_pipeline = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('regressor', rfr_obj)
                    ])

                    save_data(pd.concat([data_train_full_prediction, data_test_full_prediction]), st.session_state['data_name'], model_selection, data_to_save, task_selected, rfr_pipeline)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")

        # Option if SVR model selected
        if model_selection == 'SVM':
            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7, col8 = st.columns(4)

            with col1:
                kernel = st.radio(
                    "Kernel type",
                    ("rbf", "linear", "poly", "sigmoid", "precomputed")
                )

            with col2:
                degree = st.number_input(
                    "Degree of Polynomial kernel function",
                    value=3,
                    step=1,
                    min_value=1
                )

            with col3:
                gamma = st.radio(
                    "Kernel coefficient for 'rbf', 'poly', & 'sigmoid'",
                    ("scale", "auto")
                )

            with col4:
                tol = st.number_input(
                    "Tolerance for stopping criterion",
                    value=0.001,
                    step=0.001,
                    max_value=1.0,
                    min_value=0.00001
                )

            with col5:
                C = st.number_input(
                    "Regularization parameter",
                    value=1.0,
                    min_value=0.1,
                    step=0.1
                )

            with col6:
                epsilon = st.number_input(
                    "Epsilon value",
                    value=0.1,
                    min_value=0.001,
                    step=0.01
                )

            with col7:
                shrinking = st.radio(
                    "Shrinking heuristic",
                    (True, False)
                )

            with col8:
                cache_size = int(400)

            data_to_save['configuration']={
                "kernel":kernel,
                "degree":degree,
                "gamma":gamma,
                "tol":tol,
                "C":C,
                "epsilon":epsilon,
                "shrinking":shrinking,
                "cache_size":cache_size
            }

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # SVR Object
            svm_obj = SVR(kernel=kernel,
                          degree=degree,
                          gamma=gamma,
                          tol=tol,
                          C=C,
                          epsilon=epsilon,
                          shrinking=shrinking,
                          cache_size=cache_size
                          )

            # Fitting Data to Logistic Regression Model
            if st.button("Fit Data to SVM"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to data fitting
                    X_train = st.session_state.scaled_data_train
                    X_test = st.session_state.scaled_data_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    # Fitting model to data
                    svm_obj.fit(X_train, y_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Predicting train data
                    y_train_predict = svm_obj.predict(X_train)
                    y_train_predict_df = pd.DataFrame(y_train_predict)

                    # Predicting test data
                    y_test_predict = svm_obj.predict(X_test)
                    y_test_predict_df = pd.DataFrame(y_test_predict)

                    # Calculating mean absolute error
                    mae_train = mean_absolute_error(
                        y_train, y_train_predict)
                    mae_test = mean_absolute_error(
                        y_test, y_test_predict)

                    # Calculating mean squarred error
                    mse_train = mean_squared_error(
                        y_train, y_train_predict)
                    mse_test = mean_squared_error(
                        y_test, y_test_predict)

                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                    axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                            axis=1)
                    
                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                            y_test_df, y_test_predict_df],
                                                            axis=1)
                    data_to_save['outputs'] = {
                        "y_train_predict" : data_train_full_prediction,
                        "y_test_predict" : data_test_full_prediction,
                        "mse_train": mse_train,
                        "mse_test":mse_test,
                        "mae_train":mae_train,
                        "mae_test":mae_test
                    }
                    #show regression components
                    show_regression_metrics(data_train_full_prediction, data_test_full_prediction, mae_train, mae_test, mse_train, mse_test)

                    svm_pipeline = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('regressor', svm_obj)
                    ])

                    save_data(pd.concat([data_train_full_prediction, data_test_full_prediction]), st.session_state['data_name'], model_selection, data_to_save, task_selected, svm_pipeline)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")

    # Configuring Clustering Task
    if task_selected == "Clustering":

        # Assigning scaled_data_train key
        if "scaled_data_train" not in st.session_state:
            st.session_state['scaled_data_train'] = ""

        # Checking if scaled_data_train is DataFrame
        if type(st.session_state.scaled_data_train) == pd.DataFrame:
            st.markdown("<h3 class='menu-secondary'>Feature Data</h3>",
                    unsafe_allow_html=True)
            st.write(st.session_state.scaled_data_train)
            st.write(" - Data Shape :",
                     st.session_state.scaled_data_train.shape)

        else:
            print("")
            # Setting the upload variabel
            # uploaded_file = st.file_uploader("Choose a file to upload for training data",
            #                                  type="csv",
            #                                  help="The file will be used for training the Machine Learning",
            #                                  )

            # # Setting the upload options when there's file on uploader menu
            # if uploaded_file is not None:
            #     try:
            #         # Uploading Dataframe
            #         dataframe = get_data(uploaded_file)

            #         # Storing dataframe to session state
            #         if 'scaled_data_train' not in st.session_state:
            #             st.session_state["scaled_data_train"] = dataframe
            #         else:
            #             st.session_state.scaled_data_train = dataframe

            #         if 'data' not in st.session_state:
            #             st.session_state["data"] = dataframe
            #         else:
            #             st.session_state.data = dataframe

            #     except:
            #         st.markdown("<span class='info-box'>Please upload any data</span>",
            #                     unsafe_allow_html=True)

        # Giving two spaces
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-secondary'>Model Configuration</h3>",
                    unsafe_allow_html=True)

        # Option for Clustering Model
        model_selection = st.selectbox(
            "Select Machine Learning Model for Clustering Task",
            ("K-Means", "DBSCAN", "Agglomerative Clustering")
        )
        data_to_save['model_selection'] = model_selection
        st.write("Model selected:", model_selection)

        if model_selection == "K-Means":

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # Adding column section for K-Means Hyper-parameters
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            # Making variable of K-Means' hyper-parameter input
            with col1:
                algorithm = st.radio(
                    "K-Means Algorithm",
                    ("lloyd", "elkan"),
                    horizontal=True
                )

            with col2:
                n_clusters = st.number_input(
                    "Number of Clusters",
                    min_value=2,
                    value=3,
                    step=1
                )

            with col3:
                max_iter = st.number_input(
                    "Maximum of iterations",
                    min_value=2,
                    value=300,
                    step=1
                )
            data_to_save['configuration'] = {
                "algorithm":algorithm,
                "n_clusters":n_clusters,
                "max_iter":max_iter
            }
            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            with col4:
                init = st.radio(
                    "Method of initialization",
                    ("k-means++", "random"),
                    horizontal=True
                )

            with col5:
                n_init = st.number_input(
                    "Number of Run Different Centroid Seeds",
                    min_value=2,
                    value=10,
                    step=1
                )

            with col6:
                random_state = st.number_input(
                    "Random state",
                    min_value=1,
                    value=555,
                    step=1
                )

            # K-Means Clustering Object
            kmeans_obj = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
                algorithm=algorithm
            )

            # Fitting Data to K-Means Clustering Model
            if st.button("Fit Data to K-Means"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to fir data
                    X_train = st.session_state.scaled_data_train

                    scaler_2 = MinMaxScaler()

                    pipe_kmeans = Pipeline(steps=[("scaling", scaler_2),
                                                ("Kmeans", kmeans_obj)])

                    # Fitting data to pipeline model and getting clusters
                    clusters = pipe_kmeans.fit_predict(
                        st.session_state.scaled_data_train)

                    # Fitting data to model and getting clusters
                    # clusters = kmeans_obj.fit_predict(X_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Copy original data into new variable
                    data_full_clustered = st.session_state.data.copy()

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Added clusters into data and showing them accoridngly
                    data_full_clustered['Cluster'] = clusters
                    calinski_harabasz = calinski_harabasz_score(X_train, clusters)
                    davies_bouldin = davies_bouldin_score(X_train, clusters)

                    data_to_save['outputs']={
                        "data_full_clustered":data_full_clustered,
                        "calinski_harabasz":calinski_harabasz,
                        "davies_bouldin":davies_bouldin
                    }

                    show_clustering_metrics(data_full_clustered, calinski_harabasz, davies_bouldin)

                    kmeans_pipeline_complete = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('kmeans', pipe_kmeans)
                    ])

                    save_data(data_full_clustered, st.session_state['data_name'], model_selection, data_to_save, task_selected, kmeans_pipeline_complete)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")

        if model_selection == "DBSCAN":

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # Adding column section for DBSCAN Hyper-parameters
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            # Making variable of DBSCAN's hyper-parameter input
            with col1:
                eps = st.number_input(
                    "Maximum distance",
                    min_value=0.1,
                    value=0.5,
                    step=0.1
                )

            with col2:
                min_samples = st.number_input(
                    "Number of samples / total weight of neighborhood",
                    min_value=2,
                    value=5,
                    step=1
                )

            with col3:
                p = st.number_input(
                    "Power of Minkowski metric",
                    min_value=0,
                    value=0,
                    step=1
                )

            # Adding one spaces
                st.markdown("<br>", unsafe_allow_html=True)

            with col4:
                algorithm = st.radio(
                    "Algorithm of computing pointwise distance",
                    ("auto", "ball_tree", "kd_tree", "brute")
                )

            with col5:
                leaf_size = st.number_input(
                    "Leaf size passed to BallTree or cKDTree",
                    min_value=2,
                    value=30,
                    step=1
                )

            with col6:
                metric = st.radio(
                    "Metric of distance",
                    ("euclidean", "cityblock", "cosine", "manhattan")
                )
            data_to_save['configuration'] = {
                "algorithm":algorithm,
                "eps":eps,
                "min_samples":min_samples,
                "p":p,
                "leaf_size":leaf_size,
                "metric":metric
            }
            # K-Means Clustering Object
            dbscan_obj = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p
            )

            # Fitting Data to K-Means Clustering Model
            if st.button("Fit Data to DBSCAN"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to fir data
                    X_train = st.session_state.scaled_data_train

                    scaler_2 = MinMaxScaler()

                    pipe_dbscan = Pipeline(steps=[("scaling", scaler_2),
                                                ("DBSCAN", dbscan_obj)])

                    # Fitting data to pipeline model and getting clusters
                    clusters = pipe_dbscan.fit_predict(
                        st.session_state.scaled_data_train)

                    # Fitting data to model and getting clusters
                    # clusters = kmeans_obj.fit_predict(X_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Copy original data into new variable
                    data_full_clustered = st.session_state.data.copy()

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Added clusters into data and showing them accoridngly
                    data_full_clustered['Cluster'] = clusters
                    calinski_harabasz = calinski_harabasz_score(X_train, clusters)
                    davies_bouldin = davies_bouldin_score(X_train, clusters)

                    data_to_save['outputs']={
                        "data_full_clustered":data_full_clustered,
                        "calinski_harabasz":calinski_harabasz,
                        "davies_bouldin":davies_bouldin
                    }

                    show_clustering_metrics(data_full_clustered, calinski_harabasz, davies_bouldin)

                    dbscan_pipeline_complete = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('dbscan', pipe_dbscan)
                    ])

                    save_data(data_full_clustered, st.session_state['data_name'], model_selection, data_to_save, task_selected, dbscan_pipeline_complete)
                except:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")

        if model_selection == "Agglomerative Clustering":

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # Adding column section for Agllomerative's Hyper-parameters
            col1, col2, col3, col4 = st.columns(4)

            # Making variable of K-Means' hyper-parameter input
            with col1:
                n_clusters = st.number_input(
                    "Number of Clusters",
                    min_value=2,
                    value=2,
                    step=1
                )

            with col3:
                metric = st.radio(
                    "Method to compute linkage",
                    ("euclidean", "manhattan", "cosine", "precomputed")
                )

            with col4:
                linkage = st.radio(
                    "Linkage criterion to use",
                    ("ward", "complete", "average", "single")
                )

            data_to_save['configuration'] = {
                "n_clusters":n_clusters,
                "metric":metric,
                "linkage":linkage,
            }

            # K-Means Clustering Object
            agglomerative_obj = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=metric,
                linkage=linkage
            )

            # Fitting Data to Agglomerative Clustering Model
            if st.button("Fit Data to Agglomerative"):
                try:
                    experiment_date = str(datetime.datetime.now())
                    data_to_save['experiment_date'] = experiment_date
                    # Initiating variable to fir data
                    X_train = st.session_state.scaled_data_train

                    scaler_2 = MinMaxScaler()

                    pipe_agglomerative = Pipeline(steps=[("scaling", scaler_2),
                                                ("Agglomerative", agglomerative_obj)])

                    # Fitting data to pipeline model and getting clusters
                    clusters = pipe_agglomerative.fit_predict(
                        st.session_state.scaled_data_train)

                    # Fitting data to model and getting clusters
                    # clusters = kmeans_obj.fit_predict(X_train)

                    # Adding two spaces
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.success("Training Success")

                    # Copy original data into new variable
                    data_full_clustered = st.session_state.data.copy()

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Added clusters into data and showing them accoridngly
                    data_full_clustered['Cluster'] = clusters
                    calinski_harabasz = calinski_harabasz_score(X_train, clusters)
                    davies_bouldin = davies_bouldin_score(X_train, clusters)

                    data_to_save['outputs']={
                        "data_full_clustered":data_full_clustered,
                        "calinski_harabasz":calinski_harabasz,
                        "davies_bouldin":davies_bouldin
                    }
                    show_clustering_metrics(data_full_clustered, calinski_harabasz, davies_bouldin)

                    agglomerative_pipeline_complete = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('agglomerative', pipe_agglomerative)
                    ])

                    save_data(data_full_clustered, st.session_state['data_name'], model_selection, data_to_save, task_selected, agglomerative_pipeline_complete)
                except Exception as e:
                    st.warning("Cannot train your data, please upload your data and scale your data to train the model")