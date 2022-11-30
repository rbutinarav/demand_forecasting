##run this file to start the program
##will create an interative window with streamlit
##user will be able to explore time series and apply different algorithms

import streamlit as st
import numpy as np
import pandas as pd
from time_series_load_functions import prepare_dataset
from forecasting_metrics import evaluate
from pmdarima.arima import auto_arima

#intialize the session state variable

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    #prepare_dataset(filename='dfh_thai.csv', item='84263773',  rows='All')  ##item='All', rows='All' are deafult
    prepare_dataset(filename='demand.csv', item='All',  rows='All')  ##item='All', rows='All' are deafult
    st.session_state.first_run = False

historical_demand_monthly=st.session_state.hdm
historical_demand_yearly=st.session_state.hdy
historical_demand_quarterly=st.session_state.hdq

#demand statistics
st.write('Demand statistics')

#show historical_demand_yearly with years as columns and items as rows, order by total demand, ad a colum with total demand, first column header is item
#st.write(historical_demand_yearly.pivot_table(index='item', columns='year', values='demand', aggfunc='sum', fill_value=0, margins=True, margins_name='Total'))

#ask the user to select the item to be analyzed, order items by total demand
item = st.selectbox('Select item', historical_demand_yearly.groupby('item')['demand'].sum().sort_values(ascending=False).index)

#show the monthly demand for the selected item
st.write('Monthly demand for item', item)
st.write(historical_demand_monthly[historical_demand_monthly['item'] == item])

#plot the monthly demand, year_month as x axis, demand as y axis, x axis is categorical
st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item].set_index('year_month')['demand'])

#count the the year_months in the period for the selected item
year_months = historical_demand_monthly[historical_demand_monthly['item'] == item]['year_month'].nunique()
year_months

#ask user to define the number of periods to forecast
periods=st.slider('Number of periods to forecast', min_value=1, max_value=12, key='periods')
historical_periods=st.slider('Historical periods to analyze', min_value=1, max_value=year_months+1, key='historical_periods', value=year_months)
test_periods=st.slider('Test periods', min_value=1, max_value=year_months+1, key='test_periods', value=round(year_months*0.2))
#ask the user to choose if using ARIMA, Prophet or both
model_list = st.multiselect('Select model', ['ARIMA', 'Prophet'], default=['ARIMA','Prophet'])



#show a button to run the forecast
if st.button('Run forecast'):

    #1. PREPARE DATA FOR THE SPECIFIC PERAMETERS SELECTED BY THE USER

    #select data for the selected item
    df = historical_demand_monthly[historical_demand_monthly['item'] == item]
    #drop the column item
    df = df.drop(['item'], axis=1)
    #restrict the data to the number of historical_periods defined by the user
    df = df.tail(historical_periods)

    #convert the colum year_month to datetime64
    df['year_month'] = pd.to_datetime(df['year_month']+'01', format='%Y%m%d')
    

    if 'ARIMA' in model_list:
        #2. RUN FORECASTING USING ARIMA

        st.write('Forecasting using ARIMA')
        
        #set year_month as index
        df_ari = df.set_index('year_month')

        #split the main dataset in train and test datasets
        train_periods= (len(df_ari)-test_periods)
        train = df_ari.iloc[:train_periods]
        test = df_ari.iloc[train_periods:]

        arima_model = auto_arima   (train, #training dataset
                                    start_p=1, #number of autoregressive terms
                                    max_p=5,
                                    start_d=1, #number of nonseasonal differences
                                    max_d=2,
                                    start_q=1, #number of lagged forecast errors in the prediction equation
                                    max_q=5,
                                    seasonal=False,
                                    m=12, #periodicity of the time series
                                    n_fits=10, #number of fits to try
                                    trace=True,
                                    error_action='ignore',
                                    supress_warnings=True,
                                    stepwise=True)

        #perform predictions on the test dataset

        forecast = pd.DataFrame(arima_model.predict(n_periods=periods+test_periods))
        forecast.columns = ['demand']
        
        #create a dataframe with the actual demand and the predicted demand
        full_forecast = pd.concat([train, test, forecast], axis=1)
        #add MAPE colums
        
        full_forecast.columns = ['train', 'test', 'forecast']

        #format year_month as year-month
        full_forecast.index = full_forecast.index.strftime('%Y-%m')

        #plot the full_forecast by year_month
        st.line_chart(full_forecast)
        full_forecast

        #calculate the MAPE
        forecast_eval = full_forecast.iloc[train_periods:len(df)]

        st.write ('Valutazione forecast')
        #show only the forecast and the test columns
        st.write(forecast_eval[['forecast', 'test']])
        
        #show evaluation metrics
        evaluate_metrics = evaluate(forecast_eval['test'], forecast_eval['forecast'], metrics=('mape', 'maape'))
        evaluate_metrics
        #MAAPE https://www.sciencedirect.com/science/article/pii/S0169207016000121


    if 'Prophet' in model_list:
    ##3. RUN FORECASTING USING PROPHET

        st.write('Forecasting using Prophet')
        from prophet import Prophet

        my_model = Prophet()

        #rename column year_month into ds and column demand into y
        df_pro = df.rename(columns={'year_month': 'ds', 'demand': 'y'})
        st.write('Historical dataset', df_pro)

        # %%
        #fit the model
        my_model.fit(df_pro)

        # %%
        #create the future dates for which we want to predict the values
        future_dates = my_model.make_future_dataframe(periods, freq='M')
        st.write('Future dates',future_dates)

        # %%
        #predict the values for the next 365 days
        prediction = my_model.predict(future_dates)

        #full_forecast_pro = pd.concat([df_pro, prediction], axis=1)
        full_forecast_pro = pd.concat([prediction['ds': 'year_month', 'yhat': 'demand'],df], axis=1)
        full_forecast_pro

        #show results
        #st.write('Prophet prediction', prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])



        forecast_pro_eval = full_forecast_pro.iloc[train_periods:len(df)]

        st.write ('Valutazione forecast prophet')
        #show only the forecast and the test columns
        #forecast_pro_eval = forecast_pro_eval[['ds', 'yhat', 'y']]

        
        #show evaluation metrics
        evaluate_pro_metrics = evaluate(forecast_pro_eval['test'], forecast_pro_eval['forecast'], metrics=('mape', 'maape'))
        evaluate_pro_metrics

        #Warning: overfitting... the model is not able to predict the future values
        # https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/ 
    