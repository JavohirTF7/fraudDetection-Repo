import streamlit as st
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
import joblib


st.markdown("<h1 style = 'color: #008DDA; text-align: center; font-family: Helvetica'>FRAUD DETECTION INDICATOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFB000; text-align: center; font-family: Brush Script MT'> Built By: ILO M.A </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('40027.png', use_column_width=True)

st.header('Project Background Information', divider = True)
st.write('The primary objective of the project is to create a robust fraud detection solution capable of identifying and mitigating fraudulent transactions in real-time. By harnessing the power of machine learning algorithms, we seek to enhance the accuracy and efficiency of fraud detection while minimizing false positives. The project will utilize machine learning algorithms, including supervised learning, unsupervised learning, and anomaly detection techniques, to build predictive models for fraud detection. Historical transaction data containing both fraudulent and legitimate transactions will be used to train and validate the models. Feature engineering techniques will be applied to extract relevant features from the transaction data, such as transaction amount, location, time, and user behavior.')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('fraudTrain.csv')
st.dataframe(data.drop('Unnamed: 0', axis = 1, inplace = True))
# st.dataframe(ds.drop('trans_date_trans_time', axis = 1, inplace = True))
# st.dataframe(ds.drop('unix_time', axis = 1, inplace = True))



st.sidebar.image('pngwing.com (6).png', width = 300, caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Input User Image 
# st.sidebar.image('pngwing.com-15.png', caption = 'Welcome User')

# Apply space in the sidebar 
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)


# Declare user Input variables 
st.sidebar.subheader('Input Variables', divider= True)
new_unix= st.sidebar.number_input('new_unix_time')
amt= st.sidebar.number_input('amt')
cate_gory = st.sidebar.selectbox('category', data['category'].unique())
state = st.sidebar.selectbox('state', data['state'].unique())
mer_chant = st.sidebar.selectbox('merchant', data['merchant'].unique())
merch_long = st.sidebar.selectbox('Select Merchant Longitude', data['merch_long'].unique())
merch_lat = st.sidebar.selectbox('Select Merchant Latitude', data['merch_lat'].unique())



# display the users input
input_var = pd.DataFrame()
input_var['new_unix_time'] = [new_unix]
input_var['amt'] = [amt]
input_var['category'] = [cate_gory]
input_var['state'] = [state]
input_var['merchant'] =[mer_chant]
input_var['merch_long'] = [merch_long]
input_var['merch_lat'] = [merch_lat]



st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)

mer_chant= joblib.load('merchant_encoder.pkl')
cate_gory = joblib.load('category_encoder.pkl')
state = joblib.load('state_encoder.pkl')



#transform the users input with the imported scalers 
input_var['merchant'] = mer_chant.transform(input_var[['merchant']])
input_var['category'] =  cate_gory.transform(input_var[['category']])
input_var['state'] = state.transform(input_var[['state']])



model = joblib.load('fraudDetectionModel.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


if st.button('Predict'):
    if predicted == 1:
        st.write("Prediction: Potential fraud detected!")
        st.image("pngwing.com (4).png", caption="Potential fraud detected!", use_column_width=True)
        st.snow()
    else:
        st.write("Prediction: No fraud detected.")
        st.image("pngwing.com (5).png", caption="No fraud detected.", use_column_width=True)
        st.balloons()
