import streamlit as st
import pickle
import numpy as np
import os

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title(":red[Flipkart Laptop Price Prediction]")

# brand
company = st.selectbox('Brand Name',df['Brand Name'].unique())

# type of laptop
Os = st.selectbox('Operating System',df['OS'].unique())

# Ram
# ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
ramtype = st.selectbox('RAM Type',df['Ram Type'].unique())


ram = st.selectbox('RAM Size',df['RAM'].unique())

processor = st.selectbox('Processor',df['Processor'].unique())

gpu = st.selectbox('GPU',df['GPU'].unique())

warr = st.selectbox('Warranty',df['Warranty'].unique())

screensize = st.selectbox('Screen Size',df['ScreenSize'].unique())

disktype = st.selectbox('Disk Type',df['Disk Type'].unique())

disksize = st.selectbox('DISK SIZE',df['DISK SIZE'].unique())

if st.button('Predict Price'):
    
    query = np.array([company,Os,ramtype,ram,processor,gpu,warr,screensize,disktype,disksize])
    query = query.reshape(1,10)
    #st.title(pipe.predict(query))
    
    #st.title("Laptop Predicted Price is ₹ " + str(int(np.exp(pipe.predict(query)[0]))))
    st.title("Laptop Predicted Price is ₹ " + str(int(pipe.predict(query)[0])))
    
    # Use the pre-trained model to predict the price
    #predicted_price = int(pipe.predict(query)[0])
    # Display the predicted price to the user
    #st.success(f"Estimated laptop price is ₹ {predicted_price}")



# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # Load the pre-trained model and dataset
# pipe = pickle.load(open('pipe.pkl','rb'))
# df = pickle.load(open('df.pkl','rb'))

# # Define the page layout using Streamlit
# st.set_page_config(page_title="Flipkart Laptop Price Prediction", page_icon=":laptop:", layout="wide")
# st.title("Flipkart Laptop Price Prediction")

# # Create input widgets for user to select laptop specifications
# company = st.selectbox('Brand Name', df['Brand Name'].unique())
# os = st.selectbox('Operating System', df['OS'].unique())
# ram_type = st.selectbox('RAM Type', df['Ram Type'].unique())
# ram_size = st.selectbox('RAM Size (in GB)', df['RAM'].unique())
# processor = st.selectbox('Processor', df['Processor'].unique())
# gpu = st.selectbox('GPU', df['GPU'].unique())
# warranty = st.selectbox('Warranty', df['Warranty'].unique())
# screen_size = st.selectbox('Screen Size (in inches)', df['ScreenSize'].unique())
# disk_type = st.selectbox('Disk Type', df['Disk Type'].unique())
# disk_size = st.selectbox('DISK SIZE (in TB)', df['DISK SIZE'].unique())

# # Create a button to trigger the price prediction
# if st.button('Predict Price'):
#     # Convert the user's inputs into a query array
#     query = np.array([company, os, ram_type, ram_size, processor, gpu, warranty, screen_size, disk_type, disk_size])
#     query = query.reshape(1,10)
#     # Use the pre-trained model to predict the price
#     predicted_price = int(pipe.predict(query)[0])
#     # Display the predicted price to the user
#     st.success(f"Estimated laptop price is ₹ {predicted_price}")
