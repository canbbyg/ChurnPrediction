import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.exceptions import UndefinedMetricWarning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.combine import SMOTEENN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import warnings


def main():
    st.title("Welcome to Model Comparison App for Churn Prediction!")
    st.sidebar.title("Model Settings")

    # Suppress warnings for undefined metrics
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    @st.cache_data(persist=True)
    def load_data(uploaded_file):
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a dataset.")
            return None
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df, target_column):
        y = df[target_column]
        x = df.drop(columns=[target_column])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    def apply_balancing(x_train, y_train, method):
        if method == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif method == "ADASYN":
            sampler = ADASYN(random_state=42)
        elif method == "RUS":
            sampler = RUS(random_state=42)
        elif method == "SMOTEENN":
            sampler = SMOTEENN(random_state=42)
        else:
            return x_train, y_train
        x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)
        return x_train_resampled, y_train_resampled

    def apply_scaling(x_train, x_test, method):
        if method == "Standard Scaler":
            scaler = StandardScaler()
        elif method == "Min/Max Scaler":
            scaler = MinMaxScaler()
        else:
            return x_train, x_test
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    # File uploader widget
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    # Load the dataset
    df = load_data(uploaded_file)

    if df is None:
        st.stop()  # Stop execution if no dataset is uploaded

    # Show the dataset and allow user to select the target column
    st.write("Uploaded Dataset:")
    st.dataframe(df)
    target_column = st.sidebar.selectbox("Select the target column", options=df.columns)

    # Train-test split
    x_train, x_test, y_train, y_test = split(df, target_column)

    # Add a dropdown to apply balancing methods
    balancing_method = st.sidebar.selectbox("Balancing Method", ["None", "SMOTE", "ADASYN", "RUS", "SMOTEENN"])

    if balancing_method != "None":
        x_train, y_train = apply_balancing(x_train, y_train, balancing_method)
        st.write(f"Class distribution after {balancing_method} (training set):")
        st.write(pd.Series(y_train).value_counts())

    # Add a dropdown to select scaling method
    scaling_method = st.sidebar.selectbox("Feature Scaling Method", ["None", "Standard Scaler", "Min/Max Scaler"])

    if scaling_method != "None":
        x_train, x_test = apply_scaling(x_train, x_test, scaling_method)

    class_names = df[target_column].unique().tolist()
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Support Vector Machine (SVM)", "Random Forest (RF)", "Gradient Boosting (GBT)", "Artificial Neural Network (ANN)"))

    use_gridsearch = st.sidebar.checkbox("Use GridSearchCV for Hyperparameter Tuning?") if classifier != "Artificial Neural Network (ANN)" else False
    cv_folds = st.sidebar.number_input("Number of CV folds", 2, 10, step=1, value=3) if use_gridsearch else None

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel_svm', index=0)

        metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classify", key='classify_svm'):
            st.subheader("Support Vector Machine (SVM) Results")
            if use_gridsearch:
                param_grid = {'kernel': ['rbf', 'linear']}
                model = GridSearchCV(SVC(), param_grid, cv=cv_folds)
                st.write("Using GridSearchCV to tune hyperparameters...")
            else:
                model = SVC(kernel=kernel)

            model.fit(x_train, y_train)
            if use_gridsearch:
                st.write("Best Parameters:", model.best_params_)
                model = model.best_estimator_
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision, 2))
            st.write("Recall: ", round(recall, 2))
            plot_metrics(metrics)

    if classifier == 'Random Forest (RF)':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in forest", 100, 5000, step=10, value=500)
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, value=10)
        metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classify", key='classify_rf'):
            st.subheader("Random Forest Results")
            if use_gridsearch:
                param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 15], 'criterion': ['gini', 'entropy']}
                model = GridSearchCV(RandomForestClassifier(), param_grid, cv=cv_folds)
                st.write("Using GridSearchCV to tune hyperparameters...")
            else:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

            model.fit(x_train, y_train)
            if use_gridsearch:
                st.write("Best Parameters:", model.best_params_)
                model = model.best_estimator_
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision, 2))
            st.write("Recall: ", round(recall, 2))
            plot_metrics(metrics)

    if classifier == 'Gradient Boosting (GBT)':
        st.sidebar.subheader("Model Hyperparameters")
        learning_rate = st.sidebar.number_input("Learning rate", 0.01, 1.0, step=0.01, value=0.2)
        n_estimators = st.sidebar.number_input("Number of estimators", 100, 5000, step=10, value=200)
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, value=10)
        metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classify", key='classify_gbt'):
            st.subheader("Gradient Boosting Results")
            if use_gridsearch:
                param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10]}
                model = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=cv_folds)
                st.write("Using GridSearchCV to tune hyperparameters...")
            else:
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            model.fit(x_train, y_train)
            if use_gridsearch:
                st.write("Best Parameters:", model.best_params_)
                model = model.best_estimator_
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision, 2))
            st.write("Recall: ", round(recall, 2))
            plot_metrics(metrics)

    if classifier == 'Artificial Neural Network (ANN)':
        st.sidebar.subheader("ANN Hyperparameters")
        epochs = st.sidebar.slider("Number of Epochs", 1, 50, 20)
        batch_size = st.sidebar.slider("Batch Size", 16, 256, 32)
        metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Train ANN", key='train_ann'):
            st.subheader("Artificial Neural Network (ANN) Results")

            # Define the ANN model
            model = Sequential()
            model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))  # Input layer
            model.add(Dropout(0.2))  # Dropout
            model.add(Dense(64, activation='relu'))  # Hidden layer
            model.add(Dropout(0.2))  # Dropout
            model.add(Dense(1, activation='sigmoid'))  # Output layer

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the ANN model
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_split=0.2, verbose=1, shuffle=True)

            # Evaluate the model
            _, accuracy = model.evaluate(x_test, y_test, verbose=0)
            y_pred_prob = model.predict(x_test).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)

            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision, 2))
            st.write("Recall: ", round(recall, 2))

            # Handle metrics for ANN
            if 'Confusion Matrix' in metrics:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                fig, ax = plt.subplots()
                disp.plot(ax=ax)
                st.pyplot(fig)

            if 'ROC Curve' in metrics:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                st.pyplot(fig)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Dataset")
        st.write(df)


if __name__ == '__main__':
    main()