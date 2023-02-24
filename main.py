
##run this file to start the program
##will create an interative window with streamlit
##user will be able to explore time series and apply different algorithms

import streamlit as st
import pandas as pd
from forecasting_dataset_load import prepare_dataset
from forecasting_dataset_statistics import statistics as stat
import os

#intialize the session state variable
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = True

if 'hdm' not in st.session_state:
    st.session_state.hdm = pd.DataFrame()

if 'statistics' not in st.session_state:
    st.session_state.statistics = pd.DataFrame()

#0.0 SHOW MAIN PAGE

st.write('Demand Forecasting App')

#show a button "load a new file" to load a new file
if st.session_state.first_run == False:
    if st.sidebar.button('Load a new file'):
        st.session_state.first_run = True
        st.experimental_rerun()

if len(st.session_state.hdm) > 0:

    #add a selectbox on the left of the screen to select if showing statistics for the full dataset
    show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)

    #add a selectbox on the left to select if showing forecast details
    st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)

    #add a selectbox on the left to select if exporting logs
    st.session_state.export_logs = st.sidebar.checkbox('Export logs ', value=False)

    #add a selectbox on the left to select if exporting logs and datasets as txt files
    st.session_state.export_results = st.sidebar.checkbox('Export results ', value=False)

    #add a selectbox on the left to select if models should look for a seasonal component
    seasonal_check = st.sidebar.checkbox('Look for seasonality ', value=True)

    #add a selectbox on the left to select if models should look for a trend component
    trend_check = st.sidebar.checkbox('Look for trend ', value=True)

    #ask if keeping only items with positive demand in the last 12 months
    st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)

    #add a selectbox to select if showing items ordered by total demand or item number
    order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])


#0.1 LOAD THE DATASET

if st.session_state.first_run:
    historical_demand_monthly = prepare_dataset(filename='demand.csv', item='All',  rows='All')  ##item='All', rows='All' are deafult
    
    if historical_demand_monthly is not None:
        st.session_state.hdm = historical_demand_monthly
        st.session_state.first_run = False
        #rerun the program
        st.experimental_rerun()

if st.session_state.first_run == False:    
    historical_demand_monthly=st.session_state.hdm


#0.2 DISPLAY MAIN PAGE

if historical_demand_monthly is not None: #check if proper dataset was loaded

    #1. GET AND DISPLAY SOME STATISTICS ON THE FULL DATASET

    #get the number of total items
    total_items = historical_demand_monthly['item'].nunique()

    #get the last data
    last_date = historical_demand_monthly['year_month'].max()

    #get the fist data
    first_date = historical_demand_monthly['year_month'].min()

    #check if st.session_state.statistics is empty
    if len(st.session_state.statistics) == 0:
        st.session_state.statistics = stat(historical_demand_monthly, display=False)

    statistics = st.session_state.statistics

    if show_statistics:
        st.write('Dataset statistics')
        stat(historical_demand_monthly, statistics, display=True)


 
    #2. ASK THE USER INPUTS TO RUN THE FORECAST
    
    if st.session_state.keep_only_positive_demand:

        #from statistics, get the list of items where 'last 12 months with positive demand' is True
        items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
        historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]
            
    if order_by == 'Total demand':
        items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
    else:
        items_ordered = historical_demand_monthly['item'].unique().tolist()

    #add a searcheable selectbox to select the item to be shown
    item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)

    #add the option to skip specific items
    skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')

    #add the option to start the forecast from a specific item
    start_from_item = st.sidebar.text_input('Start from item', value='')

    #ask the user to specify the last year_month of the historical period
    last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)
    
    historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
    total_items = historical_demand_monthly['item'].nunique()



    if item_selected != '':
        if item_selected !='All':
            #show the monthly demand for the selected item
            st.write('Monthly demand for item', item_selected)
            st.write(historical_demand_monthly[historical_demand_monthly['item'] == item_selected])

            st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])

            #count the the year_months in the period for the selected item
            year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        
        else:
            #count the the year_months in the period for all items
            year_months = historical_demand_monthly['year_month'].nunique()

        #ask user to define the number of periods to forecast
        periods=st.slider('Number of periods to forecast', min_value=1, max_value=36, key='periods', value=12)
        if year_months > 1:
            historical_periods=st.slider('Historical periods to analyze', min_value=1, max_value=year_months, key='historical_periods', value=min(year_months, 48))
        else:
            historical_periods=1
        
        test_periods=st.slider('Test periods', min_value=0, max_value=year_months, key='test_periods', value=min(round(year_months*0.2),12))
        #allow to restrict the number of items to be forecasted to 1000
        if item_selected == 'All':
            number_items=st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items+1, key='number_items', value=min(total_items, 10))
        
        #ask the user to choose if using ARIMA, Prophet or both
        model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

        #show a button to run the forecast
        if st.button('Run forecast'):

            #3. RUN THE FORECAST

            #3.1. CREATE A LOG FILE AND WRITE THE PARAMETERS SELECTED BY THE USER
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #check if the log folder exists otherwise create it
            if not os.path.exists('log'):
                os.makedirs('log')
            log_file = 'log/log_' + timestamp + '.txt'
            
            with open(log_file, 'w') as f:
                f.write('Log file created at ' + timestamp + '\n')
                f.write('Parameters selected by the user: \n')
                f.write('Number of periods to forecast: ' + str(periods) + '\n')
                f.write('Historical periods to analyze: ' + str(historical_periods) + '\n')
                f.write('Test periods: ' + str(test_periods) + '\n')
                f.write('Models selected: ' + str(model_list) + '\n')
            
            #3.2 DO SOME PREPARATION BEFORE RUNNING THE FORECAST
            
            #create a progress bar object
            latest_iteration = st.empty() #create a placeholder
            progress_bar = st.progress(0)
        

            #creates the real items list based on the user input
            if item_selected == 'All':
                items = items_ordered[:number_items] #apply a cutoff based on the sorting criteria and the number of items selected by the user
                #remove the items to be skipped
                if skip_items != '':
                    items = [item for item in items if item not in skip_items.split(',')]
                if start_from_item != '':
                    items = items[items.index(start_from_item):]
            else:
                items = [item_selected]

            #inizialize the forecast and evaluation_metrics as pandas dataframes
            forecast = pd.DataFrame()
            evaluation_metrics = pd.DataFrame()


            for item in items:
                #3.3. PREPARE DATA FOR THE SPECIFIC ITEM

                #select data for the selected item
                df = historical_demand_monthly[historical_demand_monthly['item'] == item]

                #restrict to the time range selected by the user
                df = df.tail(historical_periods)

                #drop the column item
                df = df.drop(['item'], axis=1)


                #3.4. RUN THE FORECAST FOR THE SPECIFIC ITEM, TRACE LOG AND SHOW PROGRESS

                #show progress
                item_position = items.index(item)
                progress_bar.progress((item_position+1)/len(items))
                latest_iteration.text('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)))

                #log progress
                start_time = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

                if st.session_state.export_logs:
                    with open(log_file, 'a') as f:
                        f.write('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)) +': ' + item + ' started at ' + start_time +  '\n')

                #run the forecast and get the evaluation metrics
                from forecasting_compute import forecast as forecast_compute
                forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

                #add a column with the item number to the forecast and evaluation_metrics
                forecast_item['item'] = item
                evaluation_metrics_item['item'] = item

                #append forecast_item and evaluation_metrics_item to forecast and evaluation_metrics
                forecast = forecast.append(forecast_item)
                evaluation_metrics = evaluation_metrics.append(evaluation_metrics_item)

                if st.session_state.export_logs:
                    #get the timestamp for the end of the forecast
                    end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

                    #write the evaluation metrics to the log file
                    with open(log_file, 'a') as f:
                        f.write('Forecast completed at ' + end_time +  '\n')
                        f.write('Evaluation metrics: ' + str(evaluation_metrics_item) +  '\n')  

            #4. SHOW THE RESULTS

            st.balloons()

            st.write('Results')
            st.write('Number of items processed: ', len(items))

            if item_selected != 'All':
                #display data for the selected item only
                st.write ('Forecast for item', item)
                
                st.line_chart(forecast.drop(['item'], axis=1))
                if st.session_state.show_forecast_details:                
                    st.write(forecast.drop(['item'], axis=1))
                    st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))

            else:
                if st.session_state.show_forecast_details:
                    st.write('Evaluation metrics (showing first 10000 rows')
                    st.write(evaluation_metrics.head(10000))

                    st.write('Forecast (showing first 10000 rows')
                    st.write(forecast.head(10000))
                
            
                st.write('Evaluation metrics summary')
                #reset evaluation_metrics index
                evaluation_metrics_summary = evaluation_metrics.reset_index()

                if st.session_state.show_forecast_details:
                    st.write(evaluation_metrics_summary.head(10000))

                #select only the records with index = 'maape'
                evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
                #keep only the columns item and best model
                evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
                st.write(evaluation_metrics_summary)
                #show a barchart showing for each item the best model
                #groupby item and best model and count the number of records
                evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
                st.write(evaluation_metrics_summary)
                #plot a bar chart with Best model on the x axis and count on the y axis
                st.bar_chart(evaluation_metrics_summary.set_index('Best model'))
                

                #5. EXPORT THE RESULTS TO CSV FILES

                #export to a text file the forecast and the evaluation metrics
                #check if forecast_results folder exists, if not create it
                if st.session_state.export_results:                
                    if not os.path.exists('forecast_results'):
                        os.makedirs('forecast_results')

                    timestamp = pd.to_datetime('today').strftime('%Y%m%d%H%M%S')
                    forecast_csv = 'forecast_results/forecast_' + timestamp + '.csv'
                    evaluation_metrics_csv = 'forecast_results/evaluation_metrics_' + timestamp + '.csv'
    

                    #export the forecast and the evaluation metrics to csv files (with time stamp)
                    forecast.to_csv(forecast_csv, index=True)
                    evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

                    #overwrite forecast and eval files
                    #export the forecast and the evaluation metrics to csv files (without time stamp)
                    forecast.to_csv('forecast_results/forecast.csv', index=True)
                    evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

                


##IDEAS FOR IMPROVEMENTS
#additional statistics for the dataset
#show results with total forecast for each item with the best model for the period

