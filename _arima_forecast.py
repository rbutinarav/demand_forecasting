#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pmdarima.arima import auto_arima
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

arima_model = auto_arima   (train, #training dataset
                            start_p=2, #number of autoregressive terms
                            max_p=2,
                            start_d=2, #number of nonseasonal differences
                            max_d=2,
                            start_q=2, #number of lagged forecast errors in the prediction equation
                            max_q=2,
                            seasonal=True,
                            m=1, #periodicity of the time series
                            n_fits=5, #number of fits to try
                            trace=True,
                            error_action='warn',
                            supress_warnings=True,
                            stepwise=True)

arima_model.summary()

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

