def forecast(df, test_periods, periods, model_list):

    import streamlit as st
    import pandas as pd
    from forecasting_metrics import evaluate
    
    #1. PREPARE DATA
    
    #convert the colum year_month to datetime64
    df['year_month'] = pd.to_datetime(df['year_month']+'01', format='%Y%m%d')

    #set year_month as index
    df = df.set_index('year_month')

    #split the main dataset in train and test datasets
    train_periods= (len(df)-test_periods)
    train = df.iloc[:train_periods]
    test = df.iloc[train_periods:]

    #create a dataframe with the training and test datasets
    forecast = pd.concat([train, test], axis=1)
    forecast.columns = ['train', 'test']

    #st.write('This is the full forecast dataframe -debug', full_forecast)


    if 'ARIMA' in model_list:
        #2. RUN FORECASTING USING ARIMA

        from pmdarima.arima import auto_arima
        
        #build the model
        #essential parameters: y, m

        arima_model = auto_arima   (y=train, #training dataset
                                    start_p=2, #number of autoregressive terms (default p=2)
                                    max_p=5, #(default p=5)
                                    d=None, #order of differencing (default d=None)
                                    start_q=2, #order of moving average (default q=2)
                                    max_q=5, #(default q=5)
                                    start_P=1, #number of seasonal autoregressive terms (default P=1)
                                    seasonal=True, #default: True
                                    m=12, #periodicity of the time series (default =1) for monthly data m=12
                                    D=None, #order of seasonal differencing (default D=None)
                                    n_fits=10, #number of fits to try (default 10)
                                    trace=False, #default: False
                                    error_action='ignore',
                                    supress_warnings=True, #default: True
                                    stepwise=True, #default: True
                                    information_criterion='aic') #default: aic, other options: bic, aicc, oob)
        
        #ARIMA parameters explained: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
        
        #predict values for the test dataset and future periods        
        forecast_arima = pd.DataFrame(arima_model.predict(test_periods+periods))

        #add to the forecast dataframe
        forecast_arima.columns = ['ARIMA']
        forecast = pd.concat([forecast, forecast_arima], axis=1)


    if 'ETS' in model_list:
        ##4. RUN FORECASTING USING EXPONENTIAL SMOOTHING
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets
        #from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ets #this is the new version of ets
        
        #build the model
        #build a model for simple exponential smoothing
        #ets_model = ets(train, trend=None, seasonal=None).fit() #this build a model for simple exponential smoothing
        #ets_model = ets(train, trend='add', seasonal=None).fit() #this build a model for double exponential smoothing
        ets_model = ets(train, trend='add', seasonal='add').fit() #this build a model for triple exponential smoothing

        #predict values for the test dataset and future periods
        #inizialize forecast_ets dataframe
        forecast_ets = pd.DataFrame()
        forecast_ets['ETS'] = ets_model.predict(start=train_periods, end=len(df)+periods)

        #replace all negative values with 0
        forecast_ets[forecast_ets < 0] = 0

        #add to the forecast dataframe
        forecast = pd.concat([forecast, forecast_ets], axis=1)
    
    
    if 'Prophet' in model_list:
    ##5.2 RUN FORECASTING USING PROPHET - TRAINING DATASET

        from prophet import Prophet

        pro_model = Prophet()

        #create a df_pro dataframe with the columns ds and y renamed into year_month and demand
        #restore the index
        df_pro = df.reset_index(inplace=True)
        df_pro = df.rename(columns={'year_month': 'ds', 'demand': 'y'})

        #build the model
        pro_model.fit(df_pro)

        #create a dataframe with the testing and future periods
        future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
        test_and_future = future.iloc[train_periods:]
        #predict values
        forecast_pro = pro_model.predict(test_and_future)

        #create forecast_pro_renamed with only the columns ds and yhat renamed into year_month and demand
        forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
        forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')

        #replace all negative values with 0
        forecast_pro_renamed[forecast_pro_renamed < 0] = 0

        #add to the forecast dataframe
        forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)


    #if any model is selected, show the forecast and the evaluation metrics
    if len(model_list)>0:
        #st.write('Forecasting results')
        #format year_month as year-month
        forecast.index = forecast.index.strftime('%Y-%m')

        #create a dataframe restricted to the test periods
        forecast_eval = forecast.iloc[train_periods:len(df)]
        #st.write('Dataset used for evaluation', forecast_eval)

        #calculate the evaluation metrics for each model
        evaluate_metrics = {}
        for model in model_list:
            evaluate_metrics[model]=evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape','rmse', 'mse', 'mae'))
            
        #convert the dictionary into a dataframe
        evaluate_metrics = pd.DataFrame(evaluate_metrics)

        #add a column showing the best model
        evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)

        #add a column showing the metric value of the best model
        evaluate_metrics['Best model value'] = evaluate_metrics.min(axis=1)

        #show the evaluation metrics
        #st.write('Evaluation metrics:', evaluate_metrics)
        #MAAPE https://www.sciencedirect.com/science/article/pii/S0169207016000121

        #Choosing the right forecasting metric is not straightforward.
        #Let’s review the pro and con of RMSE, MAE, MAPE, and Bias. Spoiler: MAPE is the worst. Don’t use it.
        #https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d

    return forecast, evaluate_metrics
