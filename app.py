import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas_datareader.data as pdr
from keras.models import load_model
import streamlit as st
import yfinance as yfin
from datetime import date,timedelta
yfin.pdr_override()




st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.title("Figures:")
st.sidebar.markdown( "[**Table**](#data-from-2012-2022)", unsafe_allow_html=True)
st.sidebar.markdown( "[**Original**](#closing-price-vs-time-chart)", unsafe_allow_html=True)
st.sidebar.markdown( "[**100MA**](#closing-price-vs-time-chart-with-100ma)", unsafe_allow_html=True)
st.sidebar.markdown( "[**200MA**](#closing-price-vs-time-chart-with-100ma-200ma)", unsafe_allow_html=True)
st.sidebar.markdown( "[**Predictions**](#predictions-vs-original)", unsafe_allow_html=True)



st.title('Stock Trend Prediction')


user_input = st.text_input('Enter Stock Ticker',"AAPL")

yr = st.slider(label='Select a starting year', min_value=2005, max_value=2022, key=3)
start = str(yr)+"-01-01"
end = date.today() - timedelta(days = 2)

df = pdr.get_data_yahoo(user_input, start, end)

#describing data
st.text("📅 Showing data from January 1st of the selected year to today 📅")
st.subheader('Data from '+str(yr) +' - '+ str(date.today().year))
st.write(df.describe())




#visualizing data
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b',label = 'Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label = '200 Day Moving Avg')
plt.plot(df.Close,'b',label = 'Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
st.write("📊 Technical analysts in the market follow a strategy that if the 100 day MA crosses **above** the 200 day MA there is a **uptrend** 📈 and if it crosses **below** there is a **downtrend** 📉")

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label = '100 Day Moving Avg')
plt.plot(ma200,'g',label = '200 Day Moving Avg')
plt.plot(df.Close,'b',label = 'Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)




#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my lSTM model
model = load_model('keras_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0]) 

x_test, y_test = np.array(x_test), np.array(y_test)

#making predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
st.write("Utilized a Long Short Term Memory Network (LSTM) for building the model to predict the stock prices of the input stock.")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'y',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
st.progress(100)
st.write("During the creation of this project I have learned a lot about **data science and machine learning**. Here is a summary of some of the **new skills** I acquired:")
st.write(" ➡️ **Webscraping** which I used to recieve previous stock prices from yahoo finance")
st.write(" ➡️ **Data Cleansing** which I used to process and filter data to contain only what I needed")
st.write(" ➡️ Create a **machine learning** model which I utilized to predict stock prices")
st.write(" ➡️ Create graphs and visualize data using **matplotlib**")
st.write(" ➡️ Create an interactive **stream lit** web application which I used to display my model")
st.write(" ➡️ **Deploy** stream lit application to the internet using AWS EC2, S3, and Route53")
st.progress(100)
st.write("Created by **Amaan Chaudhry** 🔨")
st.write("Check out my **website** with more of my projects here: [**amaanchau.com**](http://amaanchau.com) 🖥️")

