# Streamlit app for Fetal Health Classification using a Random Forest Classifier

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


#@st.cache
#def get_data():
#    fetal_data = pd.read_csv("data/clean_fetal_data.csv")
    
#    return fetal_data

with header:
    st.title("Fetal Health Model")
    st.text("In this app we will look at fetal health. The random forest classifier from\nSklearn's library will be utilized.")


with dataset:
    st.header("Fetal health data set")
    st.text("The fetal health dataset was found within Kaggle.com.\nIt is composed of 21 total attributes.\nThe fetal_health feature classifies overall fetal health within three categories:\n1 (healthy), 2 (susceptible), and 3 (pathologic).")
    
#    fetal_data = get_data()
#    fetal_data = pd.read_csv("data/clean_fetal_data.csv") # Change data directory here
    fetal_data = pd.read_csv("data/clean_fetal_data.csv") # Change data directory here
    
    st.text("List of features in data:")
    st.write(fetal_data.columns)
    
    st.text("First five rows of data:")
    st.write(fetal_data.head(5))
    
    
    
    st.subheader('Fetal Health distribution on the fetal data set')
    fetal_health_dist = pd.DataFrame(fetal_data['fetal_health'].value_counts())
    st.bar_chart(fetal_health_dist)
    

with model_training:
    st.header("Training the model")
    st.text("Choose the hyperparameters of the model to see the changes in performance.")
    
    sel_col, disp_col = st.columns(2)
    
    criterion = sel_col.selectbox("Select the model criterion", options=['gini', 'entropy'])
    max_depth = sel_col.slider("Select the model max_depth", min_value=1, max_value=15)
    random_state = sel_col.selectbox("Select the random_state for the model", options=[0,42], index=0)

    X = fetal_data.drop(["fetal_health"], axis=1)
    y = fetal_data["fetal_health"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)
    
    rfc = RandomForestClassifier(criterion=criterion,max_depth=max_depth,random_state=random_state)
    
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    
    sel_col.subheader("The model accuracy score is: ")
    sel_col.write(accuracy_score(y_test, rfc_predict))
    
    
    
    
    
    