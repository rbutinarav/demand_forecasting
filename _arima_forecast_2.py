#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


from sklearn.metrics import r2_score as r2, mean_absolute_percentage_error as mape

#import functions from timeserie_load_prep.py
from _timeserie_load_prep import ts_load, ts_show

#%%
#execute the function ts_load() from timeserie_load_prep.py
df2 = ts_load()
print(df2.info())
ts_show(df2)

# %%
#split the main dataset in train and test datasets
train = df2.iloc[:len(df2)-365]
test = df2.iloc[len(df2)-365:]
plt.plot(train)
plt.plot(test)
# %%
#build the ARIMA model
#p: number of autoregressive terms
#d: number of nonseasonal differences
#q: number of lagged forecast errors in the prediction equation
#P, D, Q: same as p, d, q but for the seasonal component

#decomment the code below to acquire the best parameters for the model
'''
arima_model = auto_arima   (train, #training dataset
                            #start_p=1, #number of autoregressive terms
                            #max_p=3,
                            #start_d=1, #number of nonseasonal differences
                            #max_d=3,
                            #start_q=1, #number of lagged forecast errors in the prediction equation
                            #max_q=3,
                            #start_P=0, #number of autoregressive terms
                            #max_P=0,
                            #start_D=1, #number of nonseasonal differences
                            #max_D=1,
                            #start_Q=2, #number of lagged forecast errors in the prediction equation
                            #max_Q=2,                            
                            seasonal=True,
                            m=365, #periodicity of the time series
                            n_fits=5, #number of fits to try
                            trace=True,
                            error_action='warn',
                            supress_warnings=True,
                            stepwise=True)
arima_model.summary()
'''

arima_model = ARIMA(train, order=(1,1,2))
model_fit = arima_model.fit()
print (model_fit.summary())

# %%
#perform predictions on the test dataset
prediction = pd.DataFrame(arima_model.predict(n_periods=365), index=test.index)

prediction.columns = ['predicted_generation_solar']
prediction

# %%
#show the predictions for the next 365 days
plt.figure(figsize=(8,5))
plt.plot(train,label="Training", color='blue')
plt.plot(test,label="Test", color='orange')
plt.plot(prediction,label="Predicted", color='green')
plt.show()

