import streamlit as st
import pandas as pd
# import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    st.title("Mushroom Classification")
    st.sidebar.title("Mushroom Classification Web App")
    st.sidebar.write("Are your mushrooms edible or poisonous? There are three machine learning models you can explore.")
    st.write("This is a simple app to classify mushrooms into edible and poisonous.")
    
    @st.cache_data(persist=True) # Add the caching decorator
    # cache the data to avoid reloading it every time the app is run. This becomes useful when datasets are large and the app is running on a server.
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder() # LabelEncoder is to convert the categorical data into numerical data.
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    # Now we select features and perform train test split.
    def split(df):
        y = df.type
        x = df.drop(columns=["type"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 
        # random_state is to ensure the same data is used for training and testing. Any integer value can be used here.
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels = class_names) #this is how you draw plots in the new scikit learn library.
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(model, x_test, y_test, name = classifier)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, name = classifier)
            st.pyplot()


    
    df = load_data() #instantiate the function
    st.sidebar.write("   ")
    st.sidebar.write("If you wanna see the raw data, check here.")
    if st.sidebar.checkbox("Show raw data", value=False):
        st.subheader("Mushroom Data Set (Raw)")
        st.write(df) # display the raw data
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["Edible", "Poisonous"]
    st.sidebar.subheader("Choose classifier model")
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic Regression", "Random Forest"))

    if classifier == "SVM":
        st.sidebar.subheader("Tune these hyperparameters for Support Vector Machine")
        C = st.sidebar.slider("C (Regularization strength)", 0.01, 10.0, step = 0.01)
        # radio = buttons used to select one option from a list of options.
        kernel = st.sidebar.radio("Kernel (Type of decision boundary)", ("linear", "rbf"))
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"))

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 3)) #round to 3 decimal places
            st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 3)) 
            st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 3))
            st.write("F1 Score:", round(f1_score(y_test, y_pred, labels=class_names), 3))
            plot_metrics(metrics)
## F1 score is only applicable for classification problems. It is the harmonic mean of precision and recall.

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Tune these hyperparameters for Logistic Regression")
        C = st.sidebar.slider("C (Regularization strength)", 0.01, 10.0, step = 0.01, key='C_LR')
        max_iter = st.sidebar.slider("Max Iterations", 100, 500, step = 10, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 3))
            st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 3))
            st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 3))
            st.write("F1 Score:", round(f1_score(y_test, y_pred, labels=class_names), 3))
            plot_metrics(metrics)


    if classifier == "Random Forest":
        st.sidebar.subheader("Tune these hyperparameters for Random Forest")
        n_estimators = st.sidebar.slider("Number of trees", 100, 5000, step = 10, key='n_estimators')
        max_depth = st.sidebar.slider("Maximum depth of the trees", 1, 20, step = 1)
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bool(bootstrap))
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 3))
            st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 3))
            st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 3))
            st.write("F1 Score:", round(f1_score(y_test, y_pred, labels=class_names), 3))
            plot_metrics(metrics)










if __name__ == "__main__":
    main()