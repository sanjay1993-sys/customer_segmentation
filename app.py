from pycaret.clustering import *
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Final Kmeans Model')

def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df['Cluster'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('customer_segmentation.png')

    st.image(image,use_column_width=True)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to segment customer based on their behavior')
    


    st.title("Customer Segmentation Prediction App")

    if add_selectbox == 'Online':

        Age = st.number_input('Age', min_value=18, max_value=100, value=25)
        Income = st.number_input('Income', min_value=9000, max_value=200000, value=20000)
        SpendingScore = st.number_input('SpendingScore', min_value=0.0 , max_value=1.0, format="%.2f")
        Savings = st.number_input('Savings', min_value=0.0 , max_value = 25000.0, format="%.2f")


        output=""

        input_dict = {'Age' : Age, 'Income' : Income, 'SpendingScore' : SpendingScore, 'Savings' : Savings}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output =  str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()