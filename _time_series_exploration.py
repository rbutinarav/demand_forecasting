#%%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization

import os


#%%
#import CNH historical demand dataset
df = pd.read_csv('dfh_thai.csv')

#print a list of the columns of the dataset to get an idea of what we are working with
print(df.info())

#print the first 5 rows of the dataset
print(df.head())

# %%

#create a new dataset with only ITEMNUMBER, DEMANDDATE and DEMANDQUANTITY columns
df2 = df[['ITEMNUMBER', 'DEMANDDATE', 'DEMANDQUANTITY']]

# %%
#rename the columns ITEMNUMBER to item, DEMANDDATE to date, and DEMANDQUANTITY to demand
df2.columns = ['item', 'date', 'demand']

# %%

#create a new dataset filtering only the item 84263773 (CNH 4.5T Excavator)
df3 = df2[df2['item'] == '84263773']
df3.head()
#create a new dataset with date, and the sum of demand for each date
#df3 = df2.groupby('date').sum()


# %%
#alternative way to plot the dataset
plot1=df3.plot()
plot1.set_xlabel('date')
plot1.set_ylabel('demand')
plot1.axhline(df2['demand'].mean(), color='r', linestyle='dashed', linewidth=1)
plt.show()
# %%

#detect the item with the highest demand
df4 = df2.groupby('item').sum()
df4.sort_values(by=['demand'], ascending=False, inplace=True)
print(df4.head())

# %%
#reset the index of the dataset
df4 = df3.reset_index(drop=True)
#detect the time range for the dataset

dsmin= df4['date'].min()
dsmax= df4['date'].max()

#row count
rowcount = df4['date'].count()
print (dsmin, dsmax, rowcount)

