#this code will use prophet predict the values of a time series
#this is an excellent tutorial on prophet for further reference
#https://www.kaggle.com/code/prashant111/tutorial-time-series-forecasting-with-prophet/notebook'

#%%

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from prophet import Prophet

#%%

#import functions from timeserie_load_prep.py
from timeserie_load_prep import ts_load, ts_show

#%%
#execute the function ts_load() from timeserie_load_prep.py
df = ts_load()

#%%
print(df.info())
ts_show(df)
df.head()

# %%
#define the model and set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet()

#rename column date into ds and column generation solar into y
#df = df.reset_index()
df2 = df.rename(columns={'date': 'ds', 'demand': 'y'})
df2.head()

# %%
#fit the model
my_model.fit(df2)

# %%
#create the future dates for which we want to predict the values
#we want to predict the next 365 days
future_dates = my_model.make_future_dataframe(periods=365, freq='D')
future_dates.head()

# %%
#predict the values for the next 365 days
prediction = my_model.predict(future_dates)
prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
my_model.plot(prediction, uncertainty=True)
my_model.plot_components(prediction)

# %%
prediction.head()

# %%
#create a datframe with actual and predicted values
df3 = pd.DataFrame({'ds': prediction['ds'], 'yhat': prediction['yhat'], 'y': df2['y']})
df3.head()
#add MAPE column
df3['MAPE'] = np.abs((df3['y'] - df3['yhat']) / df3['y'])
df3.head(10)

# %%
#create a dataframe totalized by week
df4 = df3.groupby(pd.Grouper(key='ds', freq='W')).sum()
#add MAPE column
df4['MAPE'] = (abs(df4['y'] - df4['yhat']) / df4['y']) * 100
df4.head(10)

# %%
#create a dataframe totalize by month
df5 = df3.groupby(pd.Grouper(key='ds', freq='M')).sum()
#add MAPE column
df5['MAPE'] = (abs(df5['y'] - df5['yhat']) / df5['y']) * 100
df5.head(10)

# %%
#create a dataframe totalize by year
df6 = df3.groupby(pd.Grouper(key='ds', freq='Y')).sum()
#add MAPE column
df6['MAPE'] = (abs(df6['y'] - df6['yhat']) / df6['y']) * 100
df6.head(10)
