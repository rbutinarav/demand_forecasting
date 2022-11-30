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
from timeserie_load_prep import ts_load_full, ts_show

#%%
#execute the function ts_load_full() from timeserie_load_prep.py
df = ts_load_full()

#%%
print(df.info())
ts_show(df)
df.head()

# %%
#define the model and set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet()

#rename column date into ds and column generation solar into y
df = df.reset_index()
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
