import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model

#  Loading
model=load_model('/workspaces/Apple-Stock-Analysis/Models/LSTM.keras')

with open('/workspaces/Apple-Stock-Analysis/Models/scaler.pkl','rb') as f:
    scaler=pickle.load(f)

df=pd.read_csv('/workspaces/Apple-Stock-Analysis/Data/MSFT_data.csv',
                      index_col='Date',parse_dates=True)
n_input=60

# Prediction
def predict_price_on_date(input_date):

    target_date=pd.to_datetime(input_date)
    last_date=df.index[-1]

    # Future date
    if target_date>last_date:
        window=df['Close'].iloc[-n_input:].values.reshape(-1,1)
        window_scaled=scaler.transform(window)
        current_batch=window_scaled.reshape(1,n_input,1)

        future_dates=pd.bdate_range(start=last_date,end=target_date)
        steps=len(future_dates)

        pred_price=None
        for i in range(steps):
            pred_scaled=model.predict(current_batch,verbose=0)
            pred_price=scaler.inverse_transform(pred_scaled)[0][0]
            current_batch=np.append(
                current_batch[:, 1:, :],
                pred_scaled.reshape(1,1,1),
                axis=1
            )
        return pred_price, None   # None = no actual price for future

    # Date exists in training data
    elif target_date in df.index:
        pos=df.index.get_loc(target_date)
        window=df['Close'].iloc[pos-n_input:pos].values.reshape(-1,1)
        window_scaled=scaler.transform(window)
        X=window_scaled.reshape(1,n_input,1)

        pred_scaled=model.predict(X, verbose=0)
        pred_price=scaler.inverse_transform(pred_scaled)[0][0]
        actual_price=df['Close'].iloc[pos]
        return pred_price,actual_price

    # Weekend/holiday
    else:
        nearest=df.index[df.index.get_indexer([target_date],method='nearest')[0]]
        st.warning(f'{input_date} is not a trading day. Nearest date: {nearest.date()}')
        return predict_price_on_date(nearest)

# Streamlit
st.title('MicroSoft Stock Price ($) Predictor(Close Price)')
st.write('Select the date')

date=st.date_input('Select a date')

if st.button('Predict'):
    with st.spinner('Predicting...'):
        pred,actual=predict_price_on_date(str(date))

    st.success(f'Predicted Price: ${pred:.2f}')

    if actual is not None:
        st.info(f'Actual Price: ${actual:.2f}')
        diff=abs(actual-pred)
        st.metric(label='Difference',value=f'${diff:.2f}',
                  delta=f'{diff/actual*100:.2f}%')
    else:
        st.warning('Future date may yeil less accurate value')