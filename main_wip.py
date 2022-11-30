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

if "first_run" not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    #prepare_dataset(filename="dfh_thai.csv", item="84263773",  rows=100000)  ##item="All", rows="All" are deafult
    prepare_dataset(filename="dfh_thai.csv", item="All",  rows=10000)  ##item="All", rows="All" are deafult
    st.session_state.first_run = False

historical_demand_monthly=st.session_state.hdm
historical_demand_yearly=st.session_state.hdy
historical_demand_quarterly=st.session_state.hdq

#demand statistics
st.write("Demand statistics")

#show historical_demand_yearly with years as columns and items as rows, order by total demand, ad a colum with total demand, first column header is item
#st.write(historical_demand_yearly.pivot_table(index='item', columns='year', values='demand', aggfunc='sum', fill_value=0, margins=True, margins_name='Total'))

#ask the user to select the item to be analyzed, order items by total demand
item = st.selectbox('Select item', historical_demand_yearly.groupby('item')['demand'].sum().sort_values(ascending=False).index)

#show the monthly demand for the selected item
st.write("Monthly demand for item", item)
st.write(historical_demand_monthly[historical_demand_monthly['item'] == item])

#plot the monthly demand, year_month as x axis, demand as y axis, x axis is categorical
st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item].set_index('year_month')['demand'])

#count the the year_months in the period for the selected item
year_months = historical_demand_monthly[historical_demand_monthly['item'] == item]['year_month'].nunique()
year_months

#ask user to define the number of periods to forecast
periods=st.slider("Number of periods to forecast", min_value=1, max_value=12, key="periods")
historical_periods=st.slider("Historical periods to analyze", min_value=1, max_value=year_months+1, key="historical_periods", value=year_months)
test_periods=st.slider("Test periods", min_value=1, max_value=year_months+1, key="test_periods", value=round(year_months*0.2))

#show a button to run the forecast
if st.button("Run forecast"):

    #1. PREPARE DATA FOR THE SPECIFIC PERAMETERS SELECTED BY THE USER

    #select data for the selected item
    df = historical_demand_monthly[historical_demand_monthly['item'] == item]
    #drop the column item
    df = df.drop(['item'], axis=1)
    #restrict the data to the number of historical_periods defined by the user
    df = df.tail(historical_periods)

    #convert the colum year_month to datetime64
    df['year_month'] = pd.to_datetime(df['year_month']+'01', format='%Y%m%d')
    
    #set year_month as index
    df = df.set_index('year_month')


    #2. RUN FORECASTING USING ARIMA
    #split the main dataset in train and test datasets
    train_periods= (len(df)-test_periods)

    train = df.iloc[:train_periods]
    test = df.iloc[train_periods:]

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

    st.write ("Valutazione forecast")
    forecast_eval
    mape = np.mean(np.abs(forecast_eval['forecast'] - forecast_eval['test']) / forecast_eval['test'])

    #show MAPE
    st.write("MAPE", mape)

    
    #calculate MAAPE
    evaluate_metrics = evaluate(forecast_eval['test'], forecast_eval['forecast'], metrics=('mape', 'maape'))
    evaluate_metrics
    #MAAPE https://www.sciencedirect.com/science/article/pii/S0169207016000121


    ##3. RUN FORECASTING USING PROPHET

    from prophet import Prophet

    my_model = Prophet()

    #rename column date into ds and column generation solar into y
    df2 = df.rename(columns={'date': 'ds', 'demand': 'y'})
    df2.head()

    # %%
    #fit the model
    my_model.fit(df2)

    # %%
    #create the future dates for which we want to predict the values
    #we want to predict the next 365 days
    future_dates = my_model.make_future_dataframe(periods=366, freq='D')
    future_dates.head()

    # %%
    #predict the values for the next 365 days
    prediction = my_model.predict(future_dates)

    # %%
    #show results
    prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()